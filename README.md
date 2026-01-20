# Step3-VL-10B-Chatbot-Interface
STEP3-VL-10B Visual AI Gradio Chatbot Interface

I recommend that you download the entire collecion of files for STEP3-VL-10b: https://huggingface.co/stepfun-ai/Step3-VL-10B

You must first set the directory where your model weights are downloaded to.

model_path = "CHANGE THIS TO YOUR WEIGHTS LOCATION, FOR EXAMPLE C:\STEP3-VL-10B"

Change to: model_path = "C:\STEP3-VL-10B"

.

You can easily change how many maximum tokens can be produced by the AI by editing this part of the code:

max_new_tokens=2048,  

You can use 1024, but that tends to be quite short for the reasoning step. Other examples: 4096, 8192, etc.

Currently it appears the model chat history context history remains when you use the New Conversation button. Thus, please restart the script to start a new conversation.

License:

MIT License.
