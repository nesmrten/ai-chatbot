from typing import List
from pydantic import BaseModel

class User(BaseModel):
    id: str
    name: str

class Message(BaseModel):
    id: str
    text: str
    user_id: str
    created_at: str

class Chatroom(BaseModel):
    id: str
    name: str
    users: List[str] = []
