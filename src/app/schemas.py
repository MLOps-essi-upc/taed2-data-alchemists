from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile


class UploadImage(BaseModel):
    file: UploadFile


class DigitClass(str, Enum):
    zero = 0
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6
    seven = 7
    eight = 8
    nine = 9
