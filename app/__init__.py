from flask import Flask
from .config import Config
from .extensions import db
from .utils.logger import setup_logger
from .errors import register_error_handlers
import os

def create_app(config_class=Config):
    # Determine the absolute paths for static and templates to avoid issues
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    template_dir = os.path.join(base_dir, 'templates')
    static_dir = os.path.join(base_dir, 'static')

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)

    # Setup logger
    setup_logger(app)

    # Register error handlers
    register_error_handlers(app)

    # Register blueprints
    from .controllers.auth import auth_bp
    from .controllers.main import main_bp
    from .controllers.dashboard import dashboard_bp
    from .controllers.api import api_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)

    # Create tables
    with app.app_context():
        from . import models  # noqa: F401
        db.create_all()

    return app
