from typing import TypedDict

class user(TypedDict):
    name:str
    age:int

new_person:user={
    'name':'Aishik',
    'age':26
}

print(new_person)