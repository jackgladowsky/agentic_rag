from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from docling.document_converter import DocumentConverter
from pydantic import BaseModel
from rag_engine import RAGEngine
import os
import json
import shutil
from typing import List
import tempfile
import logging
from datetime import datetime
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from models import ChatSession, Message
from bson import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable is not set")

app = FastAPI(title="Agentic RAG Chat API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],  # Explicitly allow DELETE method
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()

# Initialize MongoDB
mongo_client = AsyncIOMotorClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
db = mongo_client.chat_database
chat_collection = db.chat_sessions

class ChatRequest(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error: str

@app.post("/chat")
async def chat(request: ChatRequest, session_id: str | None = None):
    """
    Chat endpoint that processes user messages and stores them in sessions
    
    Args:
        request: ChatRequest containing the user's message
        session_id: Optional session ID. If not provided, creates a new session
        
    Returns:
        JSON response with session_id, intermediate steps, reasoning and final answer
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Create new session if none provided
        if not session_id:
            result = await chat_collection.insert_one({
                "title": request.message[:30] + "...",  # Use first 30 chars of message as title
                "created_at": datetime.utcnow(),
                "last_updated_at": datetime.utcnow(),
                "messages": []
            })
            session_id = str(result.inserted_id)

        # Get response from RAG engine
        response = rag_engine.get_response(request.message)
        
        # Store messages in session
        await chat_collection.update_one(
            {"_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {
                                "role": "user",
                                "content": request.message,
                                "timestamp": datetime.utcnow()
                            },
                            {
                                "role": "assistant",
                                "content": response["final_answer"],
                                "timestamp": datetime.utcnow(),
                                "intermediate_steps": response.get("intermediate_steps", []),
                                "reasoning": response.get("reasoning", "")
                            }
                        ]
                    }
                },
                "$set": {"last_updated_at": datetime.utcnow()}
            }
        )
        
        # Return response with session_id
        return {
            "session_id": session_id,
            **response
        }
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload documents to the knowledge base. Supports various document types through docling parsing.
    
    Args:
        files: List of files to upload
        
    Returns:
        JSON response with status and number of files processed
    """
    try:
        processed_files = []
        converter = DocumentConverter()
        
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            
            # Create a temporary file to store the upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                # Write the uploaded file content to temp file
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()
                
                logger.info(f"Temporary file created at: {temp_file.name}")
                
                try:
                    # Use DocumentConverter to parse the document
                    result = converter.convert(temp_file.name)
                    parsed_content = result.document.export_to_markdown()
                    logger.info(f"Parsed content preview: {parsed_content[:100]}...")
                
                    # Add to ChromaDB through RAG engine
                    rag_engine.add_document(
                        content=parsed_content,
                        metadata={
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "parser": "docling"
                        }
                    )
                    
                    processed_files.append(file.filename)
                    logger.info(f"Successfully processed {file.filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing file content: {e}")
                    raise
                
                finally:
                    # Clean up temp file
                    os.unlink(temp_file.name)
        return {
            "message": f"Successfully processed {len(processed_files)} files",
            "processed_files": processed_files,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler to ensure consistent error response format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.post("/chat/{session_id}")
async def chat_with_session(session_id: str, request: ChatRequest):
    """Chat endpoint that stores messages in a session"""
    try:
        # Convert string ID to ObjectId
        response = rag_engine.get_response(request.message)
        
        # Store user message and assistant response
        await chat_collection.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {
                    "messages": {
                        "$each": [
                            {
                                "role": "user",
                                "content": request.message,
                                "timestamp": datetime.utcnow()
                            },
                            {
                                "role": "assistant",
                                "content": response["final_answer"],
                                "timestamp": datetime.utcnow(),
                                "intermediate_steps": response.get("intermediate_steps", []),
                                "reasoning": response.get("reasoning", "")
                            }
                        ]
                    }
                },
                "$set": {"last_updated_at": datetime.utcnow()}
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions():
    """Get all chat sessions"""
    try:
        sessions = await chat_collection.find().to_list(length=None)
        return [{
            "session_id": str(session["_id"]),
            "title": session.get("title", "New Chat"),
            "last_updated_at": session.get("last_updated_at", datetime.utcnow())
        } for session in sessions]
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions")
async def create_session():
    """Create a new chat session"""
    try:
        result = await chat_collection.insert_one({
            "title": "New Chat",
            "created_at": datetime.utcnow(),
            "last_updated_at": datetime.utcnow(),
            "messages": []
        })
        return {"session_id": str(result.inserted_id)}
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get a specific chat session"""
    try:
        # Convert string ID to ObjectId
        session = await chat_collection.find_one({"_id": ObjectId(session_id)})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        # Convert ObjectId to string for JSON serialization
        session["_id"] = str(session["_id"])
        return session
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-db")
async def test_db():
    try:
        # Ping the database
        await db.command("ping")
        return {"status": "Connected to MongoDB!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific chat session"""
    try:
        print(f"Received DELETE request for session: {session_id}")  # Basic print for immediate feedback
        logger.info(f"Attempting to delete session: {session_id}")
        
        if not session_id:
            logger.error("No session ID provided")
            raise HTTPException(status_code=400, detail="No session ID provided")
            
        object_id = ObjectId(session_id)
        print(f"Converting to ObjectId: {object_id}")  # Basic print
        logger.info(f"Converted to ObjectId: {object_id}")
        
        result = await chat_collection.delete_one({"_id": object_id})
        print(f"Delete result: {result.deleted_count}")  # Basic print
        logger.info(f"Delete result: {result.deleted_count}")
        
        if result.deleted_count == 0:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
            
        logger.info(f"Successfully deleted session: {session_id}")
        return {"message": "Session deleted successfully"}
    except Exception as e:
        print(f"Error in delete_session: {e}")  # Basic print
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/sessions/{session_id}/rename")
async def rename_session(session_id: str, title: str):
    """Rename a specific chat session"""
    try:
        result = await chat_collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"title": title}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return {"message": "Session renamed successfully"}
    except Exception as e:
        logger.error(f"Error renaming session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
