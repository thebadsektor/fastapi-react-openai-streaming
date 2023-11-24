import uvicorn

# Run backend and frontend with hot reload
if __name__ == "__main__":
    uvicorn.run("backend.main:app",
                host="0.0.0.0",
                port=8080,
                log_level="info",
                reload=True)