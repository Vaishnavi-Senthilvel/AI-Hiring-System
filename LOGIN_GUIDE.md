# 🔐 Login & Sign Up System - Smart Hiring Platform

## Overview
The Smart Hiring Platform now includes a comprehensive authentication system with both login and sign up capabilities to secure access to the application.

## Features

### ✅ Authentication & Authorization
- Username and password-based login
- User registration (sign up)
- Session state management
- Secure logout functionality
- User role tracking (Administrator, Recruiter, HR Manager)
- Account validation and error handling

### ✅ User Profiles
- Each user has a unique role and email
- Session information tracking
- User-specific dashboard display
- Role-based permissions

### ✅ Security Features
- Password validation (minimum 6 characters)
- Username validation (3-20 characters, alphanumeric)
- Email format validation
- Duplicate account prevention
- Session state management
- Clear error messages for failed attempts
- Logout button in sidebar for quick access

---

## Demo Login Credentials

### 👨‍💼 Administrator
- **Username:** `admin`
- **Password:** `admin123`
- **Role:** Administrator
- **Access:** Full access to all features

### 👔 Recruiter
- **Username:** `recruiter`
- **Password:** `recruiter123`
- **Role:** Recruiter
- **Access:** Resume analysis and job matching

### 👩‍💼 HR Manager
- **Username:** `hr`
- **Password:** `hr123`
- **Role:** HR Manager
- **Access:** All features

---

## How to Use

### 1. **Starting the App**
```bash
streamlit run app.py
```

### 2. **Login to Your Account**
1. Go to the **🔓 Login** tab
2. Demo credentials are displayed for reference
3. Enter your username and password
4. Click "🔓 Login" button
5. You'll be redirected to the main dashboard

### 3. **Create a New Account**
1. Go to the **📝 Sign Up** tab
2. Fill in all required fields:
   - **Username:** 3-20 characters, letters and numbers only
   - **Email:** Valid email address (e.g., user@example.com)
   - **Password:** Minimum 6 characters
   - **Confirm Password:** Must match password
   - **Role:** Choose from Recruiter, HR Manager, or Administrator
3. Click "📝 Create Account" button
4. Account is created immediately - you can log in right away!

### 4. **Logout**
- Click the logout button (🚪) in the top right of the sidebar
- You'll be returned to the login page

---

## Validation Rules

### Username Requirements
- ✅ 3-20 characters long
- ✅ Letters and numbers only (no special characters)
- ✅ Must be unique (no duplicate usernames)

### Email Requirements
- ✅ Valid email format (e.g., user@domain.com)
- ✅ Can contain dots, hyphens, underscores, and plus signs
- ✅ Must include @ symbol and domain

### Password Requirements
- ✅ Minimum 6 characters
- ✅ Maximum 50 characters
- ✅ Can include any characters

### Role Selection
- **Recruiter:** For recruitment team members
- **HR Manager:** For HR department staff
- **Administrator:** For system administrators

---

## Tab Interface

### 🔓 Login Tab
- Enter existing credentials
- Demo credentials displayed for reference
- "Help" button for getting started guide
- Direct login to main dashboard

### 📝 Sign Up Tab
- Create new user account
- Input validation for all fields
- Real-time error messages
- Easy role selection
- Immediate account activation

---

## Features in Each Module

### 📊 Dashboard
- Overview of hiring metrics
- Recent candidate matches
- Quick stats (Total Candidates, Active Positions, etc.)

### 👤 Resume Analysis
- Upload and analyze resumes
- Extract skills, education, and experience
- Detailed candidate breakdown

### 🎯 Job Matching
- Compare resumes with job descriptions
- Calculate match scores
- View skill gaps and recommendations

### 📈 Insights & Analytics
- Skill demand trends
- Top in-demand skills
- Candidate distribution by experience level

### ⚙️ Settings
- Model configuration
- Matching engine parameters
- Cache management

---

## Production Deployment

For production use, consider:

1. **Replace hardcoded credentials** with a database (MySQL, PostgreSQL, etc.)
2. **Add password hashing** using `bcrypt` or similar:
   ```python
   import bcrypt
   hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
   ```
3. **Implement JWT tokens** for session management
4. **Add email verification** for account activation
5. **Enable password reset** via email
6. **Enable SSL/TLS** for secure connections
7. **Implement rate limiting** for login attempts (e.g., 5 attempts per 15 minutes)
8. **Add audit logging** for security events
9. **Use environment variables** for configuration

### Example Production Setup:
```python
# Use environment variables and database
import os
from dotenv import load_dotenv
import psycopg2  # PostgreSQL

load_dotenv()

# Database connection
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# Password hashing
import bcrypt

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())
```

---

## Customization

### Change Demo Credentials
Edit `login.py`:
```python
USER_DATABASE = {
    "your_username": {
        "password": "your_password",
        "email": "email@example.com",
        "role": "Your Role"
    }
}
```

### Customize Login/Sign Up Pages
Edit the CSS in the `show_login_page()` function in `login.py`:
```python
st.markdown("""
    <style>
        /* Your custom CSS here */
    </style>
""", unsafe_allow_html=True)
```

### Add New Roles
1. Update signup form in `show_signup_form()`:
   ```python
   new_role = st.selectbox(
       "Role",
       ["Recruiter", "HR Manager", "Administrator", "Your New Role"],
       key="signup_role"
   )
   ```
2. Add role permissions in app.py if needed

### Modify Validation Rules
Edit validation functions in `login.py`:
```python
def validate_username(username):
    if len(username) < 3:  # Change minimum length
        return False, "Your custom message"
    # ... rest of validation
```

---

## Troubleshooting

### ❌ Can't login
- Verify username and password match the demo credentials
- Check the **🔓 Login** tab for error messages
- Try the "Help" button on the login page
- Note: Demo credentials are **admin/admin123**, **recruiter/recruiter123**, **hr/hr123**

### ❌ Can't create account
- Check all validation requirements:
  - Username must be 3-20 characters, letters and numbers only
  - Email must be valid format
  - Password must be at least 6 characters
  - Passwords must match
- Error messages will show what needs to be fixed

### ❌ Username already exists
- The username is already registered
- Choose a different username
- Or try logging in if you already have an account

### ❌ Session expires
- Streamlit reruns pages frequently
- Click "🔓 Login" again to restore your session
- Session data is maintained while the app runs

### ❌ Logout not working
- Click the 🚪 button in the top right of the sidebar
- Using other buttons will not log you out
- Session should clear and return to login page

---

## API Reference

### Core Functions

#### `verify_user(username, password)`
Verify user credentials for login
- **Parameters:** username (str), password (str)
- **Returns:** (bool, dict) - (is_valid, user_info)

#### `register_user(username, email, password, role)`
Register a new user account
- **Parameters:** username, email, password, role (all strings)
- **Returns:** (bool, str) - (success, message)

#### `validate_username(username)`
Validate username format and availability
- **Parameters:** username (str)
- **Returns:** (bool, str) - (is_valid, message)

#### `validate_email(email)`
Validate email format
- **Parameters:** email (str)
- **Returns:** (bool, str) - (is_valid, message)

#### `validate_password(password)`
Validate password strength
- **Parameters:** password (str)
- **Returns:** (bool, str) - (is_valid, message)

#### `show_login_page()`
Display login and signup interface

#### `show_logout_button()`
Display logout button in sidebar (for authenticated users)

---

## Next Steps

1. ✅ Test login with demo credentials
2. ✅ Create a new account via Sign Up tab
3. ✅ Log in with your new account
4. ✅ Explore each module (Dashboard, Resume Analysis, Job Matching)
5. ✅ Try uploading a resume (use `sample_resume.txt`)
6. 🔄 Integrate with your user database for production
7. 🔒 Add password hashing and security measures
8. 📧 Implement email verification

---

## Support

For issues or feature requests:
- Check the help information on the login page
- Review error messages for troubleshooting tips
- Contact: support@hireling.com

---

**Happy Hiring! 🚀**
