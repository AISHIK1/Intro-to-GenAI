from langchain_core.tools import tool

@tool
def multiply(a:int,b:int)->int:
    """The function multiplies 2 number and gives the product"""
    return a*b

result=multiply.invoke({'a':5,'b':10})

print(result)
print(multiply.name,multiply.args,multiply.description)