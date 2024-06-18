import random
import json


def create_room(email, cube_state):
    with open('rooms.json', 'r') as f:
        rooms = json.load(f)
        roomID = random.randint(100000, 999999)
        room = {
            "roomID": roomID,
            "user1_email": email,
            "user2_email": "",
            "cube_state": cube_state,
            "is_connected": False,
        }
        rooms.append(room)
    with open('rooms.json', 'w') as f:
        json.dump(rooms, f)
    return roomID


def join_room(email, roomID):
    with open('rooms.json', 'r') as f:
        rooms = json.load(f)
        for room in rooms:
            if room["roomID"] == int(roomID):
                room["user2_email"] = email
                room["is_connected"] = True
                cube_state = room["cube_state"]
                return cube_state
    with open('rooms.json', 'w') as f:
        json.dump(rooms, f)
    return "Room not found"


def quit_room(roomID):
    with open('rooms.json', 'r') as f:
        rooms = json.load(f)
        for room in rooms:
            if room["roomID"] == roomID:
                rooms.remove(room)
    with open('rooms.json', 'w') as f:
        json.dump(rooms, f)