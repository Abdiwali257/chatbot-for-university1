from groq_layer import polish_with_groq

question = "What is SIMAD University?"

grounded_answer = """
SIMAD University is a private higher education institution located in Mogadishu, Somalia. It was established in 1999 and provides undergraduate and postgraduate programs across multiple academic disciplines.

Source: data\\SIMAD UNIVERSITY GENERAL INFORMATION.pdf
"""

print(polish_with_groq(question, grounded_answer))