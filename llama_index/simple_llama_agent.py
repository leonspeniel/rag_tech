from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

from common.config import Config

openai = OpenAI(model="gpt-4o-mini", api_key=Config.OPENAI_API_KEY)


def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers.

    Args:
        a: Number to subtract from
        b: Number to subtract

    Returns:
        Result of a minus b
    """
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First factor
        b: Second factor

    Returns:
        Product of a and b
    """
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Quotient of a divided by b

    Raises:
        ZeroDivisionError: If divisor is zero
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


add_fun = FunctionTool.from_defaults(fn=add)
subtract_fun = FunctionTool.from_defaults(fn=subtract)
multiply_fun = FunctionTool.from_defaults(fn=multiply)
divide_fun = FunctionTool.from_defaults(fn=divide)

agent = ReActAgent.from_tools(tools=[add_fun,multiply_fun,subtract_fun,divide_fun], llm=openai, verbose=True)
response = agent.chat("what is the value of 12321+421312 and 12435351*6543?")
print(response)