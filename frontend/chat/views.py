import os

import requests
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render

from .forms import LoginForm, RegistrationForm
from .models import Chat, Message

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("chat")
    else:
        form = LoginForm()
    return render(request, "login.html", {"form": form})


def register_view(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            email = form.cleaned_data.get("email")
            password = form.cleaned_data.get("password1")

            if User.objects.filter(username=username).exists():
                form.add_error("username", "Користувач з таким іменем вже існує")
            elif form.cleaned_data.get("password1") != form.cleaned_data.get(
                "password2"
            ):
                form.add_error("password2", "Паролі не співпадають")
            else:
                user = User.objects.create_user(
                    username=username, email=email, password=password
                )
                login(request, user)
                return redirect("chat")
    else:
        form = RegistrationForm()
    return render(request, "register.html", {"form": form})


@login_required
def chat_view(request):
    chats = Chat.objects.filter(user=request.user).order_by("-created_at")
    current_chat = None
    messages = []

    chat_id = request.GET.get("chat_id")
    if chat_id:
        current_chat = get_object_or_404(Chat, id=chat_id, user=request.user)
        messages = Message.objects.filter(chat=current_chat).order_by("timestamp")

    if request.method == "POST":
        message_text = request.POST.get("message")
        chat_id = request.POST.get("chat_id")

        if not chat_id:
            current_chat = Chat.objects.create(
                user=request.user, title=message_text[:50]
            )
        else:
            current_chat = get_object_or_404(Chat, id=chat_id, user=request.user)

        Message.objects.create(
            chat=current_chat, content=message_text, is_assistant=False
        )

        try:
            response = requests.post(
                f"{BACKEND_URL}/query", json={"text": message_text}
            )

            if response.status_code == 200:
                Message.objects.create(
                    chat=current_chat,
                    content=response.json()["answer"],
                    is_assistant=True,
                )
        except requests.exceptions.RequestException as e:
            Message.objects.create(
                chat=current_chat,
                content="Sorry, I'm having trouble connecting to the server.",
                is_assistant=True,
            )

        messages = Message.objects.filter(chat=current_chat).order_by("timestamp")

    return render(
        request,
        "chat.html",
        {"chats": chats, "current_chat": current_chat, "messages": messages},
    )
