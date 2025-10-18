#!/usr/bin/env python3
"""
Simple script to run the wedding website backend server
"""
import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(backend_dir, 'backend'))

def main():
    """Run the FastAPI server"""
    try:
        import uvicorn
        from backend.main import app
        
        print("🎉 Starting Wedding Website Backend Server...")
        print("📍 Server will be available at: http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🏥 Health Check: http://localhost:8000/health")
        print("\n💡 Press Ctrl+C to stop the server\n")
        
        # Change to backend directory for proper file paths
        os.chdir(os.path.join(backend_dir, 'backend'))
        
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
        
    except ImportError as e:
        print(f"❌ Error: Missing required packages. Please install dependencies first:")
        print(f"   cd {backend_dir}/backend")
        print(f"   pip install -r requirements.txt")
        print(f"\n   Error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()