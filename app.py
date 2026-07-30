import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Ensure all necessary directories exist for production stability
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/processed', exist_ok=True)
    os.makedirs('static/reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    app.run(debug=True, port=5000, threaded=True, use_reloader=True)
