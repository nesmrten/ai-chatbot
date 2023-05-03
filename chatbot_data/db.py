from .schema import User, Message, Chatroom
from db_engine import JsonDbEngine

class ChatbotDb:
    def __init__(self, db_path: str):
        self.engine = JsonDbEngine(db_path)

    def create_user(self, user: User):
        self.engine.insert('users', user.dict())

    def get_user(self, user_id: str) -> User:
        user_data = self.engine.find('users', {'id': user_id})
        if not user_data:
            return None
        return User(**user_data[0])

    def create_message(self, message: Message):
        self.engine.insert('messages', message.dict())

    def get_messages(self, user_id: str) -> List[Message]:
        message_data = self.engine.find('messages', {'user_id': user_id}, sort_by=['-created_at'])
        return [Message(**msg) for msg in message_data]

    def create_chatroom(self, chatroom: Chatroom):
        self.engine.insert('chatrooms', chatroom.dict())

    def get_chatroom(self, chatroom_id: str) -> Chatroom:
        chatroom_data = self.engine.find('chatrooms', {'id': chatroom_id})
        if not chatroom_data:
            return None
        chatroom_dict = chatroom_data[0]
        chatroom_dict['users'] = list(chatroom_dict.get('users', []))
        return Chatroom(**chatroom_dict)

    def add_user_to_chatroom(self, user_id: str, chatroom_id: str):
        chatroom = self.get_chatroom(chatroom_id)
        if chatroom is None:
            raise ValueError(f'Chatroom {chatroom_id} not found')
        if user_id in chatroom.users:
            return
        chatroom.users.append(user_id)
        self.engine.update('chatrooms', {'id': chatroom_id}, {'users': chatroom.users})

    def remove_user_from_chatroom(self, user_id: str, chatroom_id: str):
        chatroom = self.get_chatroom(chatroom_id)
        if chatroom is None:
            raise ValueError(f'Chatroom {chatroom_id} not found')
        if user_id not in chatroom.users:
            return
        chatroom.users.remove(user_id)
        self.engine.update('chatrooms', {'id': chatroom_id}, {'users': chatroom.users})
