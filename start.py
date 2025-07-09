#!/usr/bin/env python3
"""
Simplified startup script for deployment environments
Handles missing dependencies gracefully and provides fallbacks
"""
import sys
import os
import subprocess
from pathlib import Path

def install_requirements():
    """Install requirements if needed"""
    try:
        import pandas
        import fastapi
        import uvicorn
        print("✅ Core dependencies already installed")
        return True
    except ImportError:
        print("📦 Installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            return True
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

def setup_environment():
    """Setup basic environment variables"""
    # Set default values for deployment
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent))
    os.environ.setdefault("API_HOST", "0.0.0.0")
    os.environ.setdefault("API_PORT", "8000")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("MIN_WIN_RATE", "0.80")
    
    # Use environment PORT if provided by deployment platform
    if "PORT" in os.environ:
        os.environ["API_PORT"] = os.environ["PORT"]
    
    print(f"🌐 Server will start on {os.environ['API_HOST']}:{os.environ.get('PORT', os.environ['API_PORT'])}")

def start_server():
    """Start the FastAPI server"""
    try:
        import uvicorn
        from src.api.enhanced_main import app
        
        port = int(os.environ.get("PORT", os.environ.get("API_PORT", 8000)))
        host = os.environ.get("API_HOST", "0.0.0.0")
        
        print("🚀 Starting AI-Powered Signal Engine...")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=1,  # Single worker for deployment stability
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        # Fallback: run enhanced_main.py directly
        try:
            from enhanced_main import main
            sys.argv = ["start.py", "api"]
            main()
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            sys.exit(1)

def main():
    """Main startup function"""
    print("🤖 AI Trading System - Production Startup")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Install dependencies if needed
    if not install_requirements():
        print("❌ Cannot proceed without dependencies")
        sys.exit(1)
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()