from flask import jsonify, current_app
from sqlalchemy.exc import SQLAlchemyError

def register_error_handlers(app):
    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({"error": "Unauthorized"}), 401

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({"error": "Forbidden"}), 403

    @app.errorhandler(404)
    def page_not_found(e):
        return jsonify({"error": "Page not found"}), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        current_app.logger.error(f"Internal Server Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(SQLAlchemyError)
    def handle_db_error(e):
        current_app.logger.error(f"Database Error: {str(e)}")
        return jsonify({"error": "Database error occurred"}), 500
        
    @app.errorhandler(Exception)
    def handle_exception(e):
        current_app.logger.error(f"Unhandled Exception: {str(e)}")
        # Pass through HTTP errors so they return proper status codes
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return e
        return jsonify({"error": "An unexpected error occurred"}), 500
