from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str="Aishik"
    age:Optional[int]=None
    Email: EmailStr
    cgpa:float=Field(gt=0,lt=10)

new_student={'age':32,
             'name':'Aishik Biswas',
             'Email':'aishik@example.com',
             'cgpa':8.5}

student=Student(**new_student)

print(student)