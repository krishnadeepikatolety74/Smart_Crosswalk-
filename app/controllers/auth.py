from flask import Blueprint, render_template, request, redirect, url_for, session, current_app
from ..models import User
from ..extensions import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password', 'default')
        role = request.form.get('role', 'traffic_authority')
        
        if username and not User.query.filter_by(username=username).first():
            new_user = User(username=username, role=role)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            current_app.logger.info(f"New user signed up: {username}")
        else:
            current_app.logger.warning(f"Failed signup attempt for username: {username}")
            
        return redirect(url_for('auth.login'))
    return render_template('signup.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password', 'default')
        user = User.query.filter_by(username=username).first()
        
        if user:
            if user.check_password(password):
                session['user'] = user.username
                session['role'] = user.role
                current_app.logger.info(f"User logged in: {username}")
                return redirect(url_for('main.home'))
            else:
                current_app.logger.warning(f"Invalid password for user: {username}")
                return redirect(url_for('auth.login'))
        else:
            # Fallback for simplicity matching original behavior
            current_app.logger.info(f"Fallback mock login for non-existent user: {username}")
            session['user'] = request.form.get('username', 'User')
            session['role'] = 'admin'
            return redirect(url_for('main.home'))
            
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    user = session.pop('user', None)
    session.pop('role', None)
    current_app.logger.info(f"User logged out: {user}")
    return redirect(url_for('main.landing'))
