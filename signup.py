from fastapi import APIRouter, Form, Request, status
from fastapi.templating import Jinja2Templates
from pydantic import EmailStr
from pathlib import Path
from starlette.responses import RedirectResponse
from typing import List

# --- 1. Router Setup ---
router = APIRouter()

# 🔥 IMPORTANT: Path set to 'static' folder
TEMPLATES_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Database Simulation ---
# একটি অস্থায়ী ইউজার ডেটাবেস
TEMP_USERS: List[dict] = []
user_id_counter = 1

# --- 2. Routes ---

# সাইনআপ ফর্ম দেখানোর জন্য (GET)
@router.get("/signup")
async def signup_form(request: Request):
    """Loads the signup.html page."""
    # Note: Access using /signup or /signup.html (if defined in app.py)
    return templates.TemplateResponse("signup.html", {"request": request})

# ফর্ম সাবমিশন হ্যান্ডেল করার জন্য (POST)
@router.post("/signup")
async def register_user(
    request: Request,
    fullname: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    """Handles the form submission and validation."""
    global user_id_counter

    context = {
        "request": request,
        "title": "Create Account",
        "fullname": fullname,
        "email": email
    }

    # Validation: Password Match
    if password != confirm_password:
        context["error"] = "Password and Confirm Password do not match."
        return templates.TemplateResponse("signup.html", context)
    
    # Validation: Password Length
    if len(password) < 8:
        context["error"] = "Password must be at least 8 characters long."
        return templates.TemplateResponse("signup.html", context)

    # Validation: Email Exists
    if any(u['email'] == email for u in TEMP_USERS):
        context["error"] = "Email already registered."
        return templates.TemplateResponse("signup.html", context)

    # Save User (Simulated)
    new_user = {
        "id": user_id_counter,
        "fullname": fullname,
        "email": email,
        "hashed_password": password 
    }
    TEMP_USERS.append(new_user)
    user_id_counter += 1

    # Success Page Redirection/Rendering
    return templates.TemplateResponse(
        "success.html", 
        {"request": request, "title": "Registration Successful", "user_name": fullname}
    )

# --- Optional Route to check database state ---
@router.get("/debug/users")
def get_temp_users():
    return TEMP_USERS