"""
Login & Sign Up Module for Smart Hiring Platform
"""
import streamlit as st
import hashlib
from datetime import datetime

# User credentials database (for demo - in production use proper authentication)
USER_DATABASE = {
    "admin": {
        "password": "admin123",
        "email": "admin@hireling.com",
        "role": "Administrator"
    },
    "recruiter": {
        "password": "recruiter123",
        "email": "recruiter@hireling.com",
        "role": "Recruiter"
    },
    "hr": {
        "password": "hr123",
        "email": "hr@hireling.com",
        "role": "HR Manager"
    }
}

def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(str(password).encode()).hexdigest()

def verify_user(username, password):
    """Verify user credentials"""
    if username in USER_DATABASE:
        user = USER_DATABASE[username]
        if user["password"] == password:
            return True, user
    return False, None

def validate_username(username):
    """Validate username format and availability"""
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    if len(username) > 20:
        return False, "Username must be less than 20 characters"
    if not username.isalnum():
        return False, "Username can only contain letters and numbers"
    if username in USER_DATABASE:
        return False, "Username already exists"
    return True, "Valid"

def validate_email(email):
    """Validate email format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True, "Valid"
    return False, "Invalid email format"

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if len(password) > 50:
        return False, "Password must be less than 50 characters"
    return True, "Valid"

def register_user(username, email, password, role):
    """Register a new user"""
    # Validate inputs
    is_valid, msg = validate_username(username)
    if not is_valid:
        return False, msg
    
    is_valid, msg = validate_email(email)
    if not is_valid:
        return False, msg
    
    is_valid, msg = validate_password(password)
    if not is_valid:
        return False, msg
    
    # Add user to database
    USER_DATABASE[username] = {
        "password": password,
        "email": email,
        "role": role
    }
    
    return True, "User registered successfully! Please log in."

def show_login_page():
    """Display login and signup page"""
    st.set_page_config(
        page_title="Smart Hiring Platform - Login",
        page_icon="🚀",
        layout="centered"
    )
    
    # Custom CSS for login page
    st.markdown("""
        <style>
            .login-container {
                max-width: 400px;
                margin: 0 auto;
            }
            .login-header {
                text-align: center;
                color: #2E86AB;
                margin-bottom: 30px;
            }
            .login-box {
                background-color: #F0F2F6;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .demo-info {
                background-color: #E3F2FD;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #2E86AB;
            }
            .demo-info h4 {
                margin-top: 0;
                color: #1565C0;
            }
            .demo-cred {
                font-family: monospace;
                font-size: 12px;
                color: #333;
                margin: 5px 0;
            }
            .tab-buttons {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .tab-button {
                flex: 1;
                padding: 10px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
            }
            .tab-active {
                background-color: #2E86AB;
                color: white;
            }
            .tab-inactive {
                background-color: #E0E0E0;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class="login-header">
                <h1>🚀 Smart Hiring Platform</h1>
                <p>AI-Powered Candidate Analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize state for tab
        if 'auth_tab' not in st.session_state:
            st.session_state.auth_tab = "login"
        
        # Tab selection
        tab1, tab2 = st.tabs(["🔓 Login", "📝 Sign Up"])
        
        with tab1:
            show_login_form()
        
        with tab2:
            show_signup_form()
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
            <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
                <p>© 2024 Smart Hiring Platform. All rights reserved.</p>
                <p>For support, contact: support@hireling.com</p>
            </div>
        """, unsafe_allow_html=True)

def show_login_form():
    """Display login form"""
    st.markdown("""
        <div class="demo-info">
            <h4>📝 Demo Credentials</h4>
            <div class="demo-cred"><strong>Username:</strong> admin</div>
            <div class="demo-cred"><strong>Password:</strong> admin123</div>
            <hr style="margin: 10px 0;">
            <div class="demo-cred"><strong>Username:</strong> recruiter</div>
            <div class="demo-cred"><strong>Password:</strong> recruiter123</div>
            <hr style="margin: 10px 0;">
            <div class="demo-cred"><strong>Username:</strong> hr</div>
            <div class="demo-cred"><strong>Password:</strong> hr123</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🔐 Login to Your Account")
    
    username = st.text_input(
        "Username",
        placeholder="Enter your username",
        key="login_username"
    )
    
    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password",
        key="login_password"
    )
    
    # Login button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔓 Login", use_container_width=True, key="login_btn"):
            if not username or not password:
                st.error("❌ Please enter both username and password")
            else:
                is_valid, user_info = verify_user(username, password)
                
                if is_valid:
                    # Set session state
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_info = user_info
                    st.session_state.login_time = datetime.now()
                    
                    st.success(f"✅ Welcome, {user_info['role']}!")
                    st.balloons()
                    
                    # Redirect to main app
                    import time
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
                    st.info("💡 Use the demo credentials above to log in, or create a new account")
    
    with col2:
        if st.button("ℹ️ Help", use_container_width=True, key="login_help"):
            st.info("""
                ### Getting Started
                - Use the demo credentials provided
                - Or create a new account in the Sign Up tab
                - Each role has different permissions
            """)

def show_signup_form():
    """Display signup form"""
    st.subheader("📝 Create New Account")
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        new_username = st.text_input(
            "Username",
            placeholder="Choose a username",
            key="signup_username",
            help="3-20 characters, letters and numbers only"
        )
    
    with col2:
        new_email = st.text_input(
            "Email",
            placeholder="your.email@example.com",
            key="signup_email"
        )
    
    new_password = st.text_input(
        "Password",
        type="password",
        placeholder="Create a password (min 6 characters)",
        key="signup_password"
    )
    
    confirm_password = st.text_input(
        "Confirm Password",
        type="password",
        placeholder="Confirm your password",
        key="signup_confirm_password"
    )
    
    new_role = st.selectbox(
        "Role",
        ["Recruiter", "HR Manager", "Administrator"],
        key="signup_role",
        help="Select your role in the platform"
    )
    
    st.markdown("---")
    
    # Sign up button
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Create Account", use_container_width=True, key="signup_btn"):
            # Validate inputs
            if not new_username or not new_email or not new_password or not confirm_password:
                st.error("❌ Please fill in all fields")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match")
            else:
                # Register user
                success, message = register_user(new_username, new_email, new_password, new_role)
                
                if success:
                    st.success(f"✅ {message}")
                    st.info(f"Your new account:\n- **Username:** {new_username}\n- **Email:** {new_email}\n- **Role:** {new_role}")
                    st.success("🎉 You can now log in with your credentials!")
                else:
                    st.error(f"❌ Registration failed: {message}")
    
    with col2:
        if st.button("ℹ️ Info", use_container_width=True, key="signup_help"):
            st.info("""
                ### Account Requirements
                - **Username:** 3-20 characters, letters and numbers only
                - **Email:** Valid email address
                - **Password:** At least 6 characters
                - **Role:** Choose your role in the platform
                
                After registering, you can log in immediately.
            """)
    
    st.markdown("---")
    
    st.markdown("""
        <div style="background-color: #F0F9FF; padding: 15px; border-radius: 8px; border-left: 4px solid #0EA5E9;">
            <p style="margin: 0; font-size: 14px; color: #0369A1;">
                <strong>ℹ️ Note:</strong> This is a demo version. In production, passwords would be encrypted and stored securely in a database.
            </p>
        </div>
    """, unsafe_allow_html=True)

def show_logout_button():
    """Show logout button in sidebar"""
    if st.session_state.get('logged_in'):
        user_info = st.session_state.get('user_info', {})
        
        with st.sidebar:
            st.markdown("---")
            
            # User info
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                        <div style="padding: 10px;">
                            <p style="margin: 0; font-weight: bold;">👤 {st.session_state.username}</p>
                            <p style="margin: 5px 0; font-size: 12px; color: #666;">{user_info.get('role', 'User')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("🚪", help="Logout", key="logout_btn"):
                        # Clear session state
                        st.session_state.logged_in = False
                        st.session_state.username = None
                        st.session_state.user_info = None
                        st.session_state.login_time = None
                        
                        st.info("✅ Successfully logged out")
                        import time
                        time.sleep(1)
                        st.rerun()
