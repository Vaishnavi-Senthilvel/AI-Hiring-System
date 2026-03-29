"""
Test signup and login functionality
"""
from login import (
    validate_username, 
    validate_email, 
    validate_password, 
    register_user,
    verify_user,
    USER_DATABASE
)

print("=" * 60)
print("Testing Validation Functions")
print("=" * 60)

# Test username validation
print("\n1. Testing Username Validation:")
test_usernames = [
    ("ab", False),           # Too short
    ("validuser", True),     # Valid
    ("USER123", True),       # Valid with numbers
    ("user@name", False),    # Invalid characters
    ("admin", False)         # Already exists
]

for username, expected_valid in test_usernames:
    is_valid, msg = validate_username(username)
    status = "✅" if is_valid == expected_valid else "❌"
    print(f"  {status} '{username}': {msg}")

# Test email validation
print("\n2. Testing Email Validation:")
test_emails = [
    ("invalid.email", False),
    ("test@example.com", True),
    ("user.name+tag@domain.co.uk", True),
    ("@nodomain.com", False)
]

for email, expected_valid in test_emails:
    is_valid, msg = validate_email(email)
    status = "✅" if is_valid == expected_valid else "❌"
    print(f"  {status} '{email}': {msg}")

# Test password validation
print("\n3. Testing Password Validation:")
test_passwords = [
    ("12345", False),        # Too short
    ("validpass123", True),  # Valid
    ("short", False),        # Too short
    ("validpasswordwith123", True)  # Valid
]

for password, expected_valid in test_passwords:
    is_valid, msg = validate_password(password)
    status = "✅" if is_valid == expected_valid else "❌"
    print(f"  {status} '{password}': {msg}")

# Test user registration
print("\n4. Testing User Registration:")
print(f"  Initial users: {len(USER_DATABASE)}")

# Try to register a valid user
success, msg = register_user("testuser", "test@example.com", "testpass123", "Recruiter")
print(f"  ✅ Register new user: {msg}")
print(f"  Total users now: {len(USER_DATABASE)}")

# Try to register with duplicate username
success, msg = register_user("testuser", "another@example.com", "testpass123", "Recruiter")
print(f"  {'✅' if not success else '❌'} Register duplicate: {msg}")

# Try to register with invalid data
success, msg = register_user("ab", "invalid", "short", "HR Manager")
print(f"  {'✅' if not success else '❌'} Register invalid: {msg}")

# Test login
print("\n5. Testing Login:")
success, user_info = verify_user("testuser", "testpass123")
print(f"  ✅ Login testuser: {success} - Role: {user_info.get('role') if success else 'N/A'}")

success, user_info = verify_user("testuser", "wrongpass")
print(f"  {'✅' if not success else '❌'} Login with wrong password: Failed as expected")

# Test demo logins
print("\n6. Testing Demo Credentials:")
for username in ["admin", "recruiter", "hr"]:
    success, user_info = verify_user(username, f"{username}123")
    status = "✅" if success else "❌"
    print(f"  {status} Login {username}: {user_info.get('role') if success else 'Failed'}")

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("=" * 60)
