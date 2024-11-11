import os
from zhipuai import ZhipuAI

# Here you need to replace the API Key and API Secret with your，I provide a test key and secret here

client = ZhipuAI(api_key = '79ae6ffbd0565c9b631ad55bccb20bd4.4g6C2x2ui7cU3Xh5')

response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {
            "role": "user",
            "content": "tell me a joke"
        }
    ],
    top_p=0.9,
    temperature=0.7,
    stream=False,
    max_tokens=2000,
    do_sample=True, # set False to use greedy decoding like temperature=0.0
)
print(response)