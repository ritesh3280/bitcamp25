"""
Pinecone Vector Database Utilities for Frame Analysis

This module provides functions to work with Pinecone vector database
for semantic search across video frames.
"""

import os
import json
import time
from openai import OpenAI
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize connections
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "framevectors"

# Create index if it doesn't exist
try:
    # Check if our index already exists
    indexes = pc.list_indexes()
    
    if index_name not in [index.name for index in indexes]:
        print(f"Creating new Pinecone index: {index_name}")
        
        # Create the index with appropriate dimensions for Ada embeddings
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimension for text-embedding-ada-002
            metric="cosine"
        )
        # Wait for index to be created
        time.sleep(1)
except Exception as e:
    print(f"Error setting up Pinecone index: {e}")

# Get the index
try:
    index = pc.Index(index_name)
except Exception as e:
    print(f"Error connecting to Pinecone index: {e}")
    index = None

def create_searchable_text(frame):
    """Create searchable text from frame data"""
    # Combine detected objects
    objects_text = ', '.join(frame.get('objects', []))
    
    # Get description if available
    description = frame.get('llm_description', {}).get('description', '')
    
    # Combine relevant fields
    search_fields = [
        objects_text,
        description,
        frame.get('timestamp', '')
    ]
    
    return ' '.join(filter(bool, search_fields))

def extract_frame_highlights(frame):
    """Extract key highlights from a frame to display in search results without LLM calls"""
    # Get description if available
    description = frame.get('llm_description', {}).get('description', '')
    
    # If there's a description, use the first sentence
    if description:
        # Get first sentence or up to 100 chars
        highlight = description.split('. ')[0]
        if len(highlight) > 100:
            highlight = highlight[:97] + "..."
        return highlight
    
    # If no description, create a simple summary from objects
    objects = frame.get('objects', [])
    if objects:
        if len(objects) > 3:
            return f"Frame contains {objects[0]}, {objects[1]}, {objects[2]}, and {len(objects)-3} more objects"
        else:
            return f"Frame contains {', '.join(objects)}"
    
    # Fallback
    return "Frame with no detailed description available"

def upsert_session_vectors(session_id, frames):
    """Upsert frame vectors for a session to Pinecone"""
    if not index:
        print("Pinecone index not available")
        return False
    
    # Check if this session has already been vectorized
    try:
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        
        # If the namespace exists and has vectors, we can skip
        if session_id in namespaces and namespaces[session_id]["vector_count"] > 0:
            print(f"Session {session_id} already vectorized with {namespaces[session_id]['vector_count']} vectors")
            return True
    except Exception as e:
        print(f"Error checking index stats: {e}")
    
    vectors_to_upsert = []
    print(f"Vectorizing {len(frames)} frames for session {session_id}")
    
    for frame in frames:
        # Skip frames without LLM descriptions
        if not frame.get('llm_description'):
            continue
            
        search_text = create_searchable_text(frame)
        
        # Skip if there's not enough meaningful text to embed
        if len(search_text.strip()) < 10:
            continue
            
        try:
            # Generate embedding
            vector_response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=search_text
            )
            vector = vector_response.data[0].embedding
            
            # Extract frame highlights (no LLM call)
            highlight = extract_frame_highlights(frame)
            
            # Create vector record
            vectors_to_upsert.append({
                "id": frame.get('frame_id'),
                "values": vector,
                "metadata": {
                    "frame_id": frame.get('frame_id'),
                    "timestamp": frame.get('timestamp', ''),
                    "objects": frame.get('objects', []),
                    "description": frame.get('llm_description', {}).get('description', ''),
                    "highlight": highlight,
                    "confidence": max(frame.get('confidence', {}).values()) if frame.get('confidence') else 0,
                    "session_id": session_id,
                    "full_text": search_text
                }
            })
        except Exception as e:
            print(f"Error creating embedding for frame {frame.get('frame_id')}: {e}")
    
    # Upsert vectors in batches
    if vectors_to_upsert:
        try:
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
                index.upsert(vectors=batch, namespace=session_id)
            print(f"Upserted {len(vectors_to_upsert)} frame vectors for session {session_id}")
            return True
        except Exception as e:
            print(f"Error upserting vectors: {e}")
            return False
    else:
        print(f"No vectors to upsert for session {session_id}")
        return False

def semantic_search_frames(session_id, query, top_k=5):
    """Search for semantically similar frames in a session using Pinecone"""
    if not index:
        raise Exception("Pinecone index not available")
    
    try:
        # Generate embedding for query - THIS IS THE ONLY LLM CALL WE MAKE
        query_embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        
        # Search Pinecone
        results = index.query(
            namespace=session_id,
            vector=query_embedding.data[0].embedding,
            top_k=top_k * 2,  # Get more results than needed to filter
            include_metadata=True
        )
        
        # Process results
        matches = []
        for match in results.get('matches', []):
            # Skip low similarity results
            if match.get('score', 0) < 0.6:
                continue
                
            metadata = match.get('metadata', {})
            frame_id = metadata.get('frame_id')
            
            # Use the pre-extracted highlight instead of generating an explanation
            explanation = metadata.get('highlight', 'Relevant frame found')
            
            matches.append({
                "id": frame_id,
                "similarity": match.get('score', 0),
                "timestamp": metadata.get('timestamp', ''),
                "objects": metadata.get('objects', []),
                "explanation": explanation,
                "confidence": metadata.get('confidence', 0)
            })
        
        # Limit to top_k results
        return matches[:top_k]
    except Exception as e:
        print(f"Error in semantic search: {e}")
        raise e

def delete_session_vectors(session_id):
    """Delete all vectors for a session from Pinecone"""
    if not index:
        print("Pinecone index not available")
        return False
    
    try:
        # Delete all vectors in the namespace
        index.delete(delete_all=True, namespace=session_id)
        print(f"Deleted all vectors for session {session_id}")
        return True
    except Exception as e:
        print(f"Error deleting vectors for session {session_id}: {e}")
        return False

def delete_all_vectors():
    """Delete all vectors from all namespaces"""
    if not index:
        print("Pinecone index not available")
        return False
    
    try:
        # Get all namespaces
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {}).keys()
        
        # Delete each namespace
        for namespace in namespaces:
            index.delete(delete_all=True, namespace=namespace)
            print(f"Deleted all vectors for namespace {namespace}")
        
        return True
    except Exception as e:
        print(f"Error deleting all vectors: {e}")
        return False 