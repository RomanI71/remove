from fastapi import APIRouter, Form, Request, status
from fastapi.templating import Jinja2Templates
from pydantic import EmailStr
from pathlib import Path
from starlette.responses import RedirectResponse
from typing import List
from fastapi.templating import Jinja2Templates

# --- 1. Router Setup ---
router = APIRouter()

# টেমপ্লেট ফোল্ডার কনফিগারেশন (আপনার HTML ফাইলগুলো static ফোল্ডারে আছে)
TEMPLATES_DIR = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates = Jinja2Templates(directory="templates")

# --- 2. Simulated Database Access ---
# NOTE: বাস্তব অ্যাপ্লিকেশনে, আপনি এখানে একটি মডিউল থেকে
# TEMP_USERS বা ডাটাবেস সংযোগটি import করবেন যাতে signup.py
# এবং login.py একই ব্যবহারকারীর ডেটা অ্যাক্সেস করতে পারে।

# আপাতত ডেমোর জন্য কিছু ইউজার ডেটা সেট করা হলো:
TEMP_USERS_DUMMY = [
    {"email": "test@example.com", "password": "securepassword123", "fullname": "Test User"},
]

# আপনার যদি signup.py তে TEMP_USERS নামে একটি তালিকা থাকে,
# তবে login.py তে সেই তালিকাটি import করে ব্যবহার করতে হবে।

# --- 3. Routes ---

# লগইন ফর্ম দেখানোর জন্য (GET Request)
@router.get("/login")
async def login_form(request: Request):
    """Loads the login.html page."""
    # আমরা /static/login.html এর পরিবর্তে /login রুট ব্যবহার করব
    return templates.TemplateResponse("login.html", {"request": request, "title": "Log In"})

# লগইন সাবমিশন হ্যান্ডেল করার জন্য (POST Request)
@router.post("/login")
async def login_user(
    request: Request,
    email: EmailStr = Form(...),
    password: str = Form(...)
):
    """Handles the user authentication."""
    
    # context এরর মেসেজ দেখানোর জন্য
    context = {
        "request": request,
        "title": "Log In",
        "email": email
    }
    
    # 🔥 ১. ডাটাবেসে ইউজার চেক করা (বাস্তবে hashed password চেক হবে)
    user_found = None
    # NOTE: এখানে TEMP_USERS_DUMMY এর পরিবর্তে আপনার আসল user list ব্যবহার করুন
    for user in TEMP_USERS_DUMMY:
        if user["email"] == email:
            user_found = user
            break

    if not user_found or user_found["password"] != password:
        context["error"] = "Invalid email or password."
        return templates.TemplateResponse("login.html", context)

    # 🔥 ২. সফল হলে অন্য পেজে রিডাইরেক্ট বা সাকসেস মেসেজ
    # সাধারণত সফল লগইনের পর ইউজারকে সেশন/টোকেন দিয়ে হোমপেজে পাঠানো হয়।
    # আমরা আপাতত ড্যাশবোর্ড.html পেজে পাঠাচ্ছি।
    response = templates.TemplateResponse(
        "dashboard.html", 
        {"request": request, "title": "Dashboard", "user_name": user_found["fullname"]}
    )
    
    # ঐচ্ছিক: উদাহরণস্বরূপ একটি ডামি কুকি সেট করা হলো
    response.set_cookie(key="auth_token", value="some_jwt_token_here", httponly=True)
    return response

