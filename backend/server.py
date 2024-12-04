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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()

class ChatRequest(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes user messages and returns reasoned responses
    
    Args:
        request: ChatRequest containing the user's message
        
    Returns:
        JSON response with intermediate steps, reasoning and final answer
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        response = rag_engine.get_response(request.message)
        
        # Return the response directly since it's already in the correct format
        return response
        
    except Exception as e:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
