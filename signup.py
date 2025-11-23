from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from typing import List
from pathlib import Path

# --- 1. Router Setup (App এর বদলে Router) ---
router = APIRouter()

# টেমপ্লেট ফোল্ডার চিনিয়ে দেওয়া (static ফোল্ডারের ভেতরে থাকলে)
# আপনার app.py এর ফোল্ডার স্ট্রাকচার অনুযায়ী এটি 'static' বা 'templates' হতে পারে
TEMPLATES_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- 2. Database Simulation ---
fake_db: List[dict] = []
user_id_counter = 1

# --- 3. Routes ---

# ১. সাইন আপ পেজ দেখানোর জন্য (GET Request)
@router.get("/signup.html")
async def signup_page(request: Request):
    return templates.TemplateResponse(
        "signup.html",
        {"request": request, "title": "Create Account"}
    )

# ২. ফর্ম সাবমিট করার জন্য (POST Request)
@router.post("/signup")
async def register_user(
    request: Request,
    # HTML Form থেকে ডেটা নেওয়ার জন্য Form(...) ব্যবহার করা হয়েছে
    fullname: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...)
):
    global user_id_counter

    # এরর বা সাকসেস মেসেজ দেখানোর জন্য কনটেক্সট
    context = {
        "request": request,
        "title": "Create Account",
        "fullname": fullname,
        "email": email
    }

    # --- Validation Logic ---
    if password != confirm_password:
        context["error"] = "Password and Confirm Password do not match."
        # এরর হলে আবার সাইন আপ পেজেই ফেরত পাঠানো হবে এরর মেসেজসহ
        return templates.TemplateResponse("signup.html", context)
    
    if len(password) < 8:
        context["error"] = "Password must be at least 8 characters long."
        return templates.TemplateResponse("signup.html", context)

    # ইমেইল চেক করা
    if any(u['email'] == email for u in fake_db):
        context["error"] = "Email already registered."
        return templates.TemplateResponse("signup.html", context)

    # --- Save User ---
    new_user = {
        "id": user_id_counter,
        "fullname": fullname,
        "email": email,
        "hashed_password": password 
    }
    fake_db.append(new_user)
    user_id_counter += 1

    # সফল হলে success.html পেজ দেখানো
    return templates.TemplateResponse(
        "success.html",
        {"request": request, "title": "Registration Successful", "user_name": fullname}
    )