import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

# Load model and processor
model_path = "CHANGE THIS TO YOUR WEIGHTS LOCATION, FOR EXAMPLE C:\STEP3-VL-10B"
key_mapping = {
    "language_model.lm_head.weight": "language_model.model.embed_tokens.weight"
}

# CORRECT key_mapping from official docs
key_mapping = {
    "^vision_model": "model.vision_model",
    r"^model(?!\.(language_model|vision_model))": "model.language_model",
    "vit_large_projector": "model.vit_large_projector",
}

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    model_path, 
    trust_remote_code=True,
    local_files_only=True,
    fix_mistral_regex=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    dtype="auto",
    key_mapping=key_mapping,
    local_files_only=True
).eval()
print("")
print("Model loaded successfully!")
print("")

# Conversation state
conversation_history = []
current_image = None

def process_message(user_message, image, history):
    """Process user message and generate response"""
    global conversation_history, current_image
    
    if image is not None:
        current_image = image
    
    # Build message content
    content = []
    if current_image is not None:
        content.append({"type": "image", "image": current_image})
    content.append({"type": "text", "text": user_message})
    
    # Add to conversation history
    conversation_history.append({
        "role": "user",
        "content": content
    })
    
    # Prepare input
    inputs = processor.apply_chat_template(
        conversation_history,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    response = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    
    # Add assistant response to history
    conversation_history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response}]
    })
    
    # Format for Gradio chatbot (messages format)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    
    return history, ""


def clear_image():
    """Clear the current image"""
    global current_image
    current_image = None
    return None

def new_conversation():
    """Start a new conversation"""
    global conversation_history, current_image
    conversation_history = []
    current_image = None
    return [], None, ""

# Custom CSS for black and purple theme
custom_css = """
#main-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0a1f 100%);
}

.gradio-container {
    background: #0a0a0a !important;
}

#chatbot {
    background: #1a0a2e !important;
    border: 2px solid #6b21a8 !important;
    border-radius: 12px !important;
}

#chatbot .message.user {
    background: linear-gradient(135deg, #6b21a8 0%, #9333ea 100%) !important;
    color: white !important;
    border-radius: 12px !important;
}

#chatbot .message.bot {
    background: #16213e !important;
    color: #e9d5ff !important;
    border: 1px solid #6b21a8 !important;
    border-radius: 12px !important;
}

.input-box, textarea {
    background: #1a1a2e !important;
    border: 2px solid #6b21a8 !important;
    color: #e9d5ff !important;
    border-radius: 8px !important;
}

button {
    background: linear-gradient(135deg, #6b21a8 0%, #9333ea 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    transition: transform 0.2s !important;
}

button:hover {
    transform: scale(1.05) !important;
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
}

.clear-btn {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
}

.clear-btn:hover {
    background: linear-gradient(135deg, #f87171 0%, #ef4444 100%) !important;
}

h1 {
    color: #c084fc !important;
    text-align: center !important;
    font-size: 2.5em !important;
    text-shadow: 0 0 20px #9333ea !important;
}

label {
    color: #e9d5ff !important;
    font-weight: bold !important;
}

#image-preview {
    border: 2px solid #6b21a8 !important;
    border-radius: 12px !important;
    background: #16213e !important;
}
"""

# Build Gradio interface
with gr.Blocks(css=custom_css, title="STEP3-VL-10B Chat") as demo:
    gr.Markdown("# 🎨 STEP3-VL-10B Vision Chat", elem_id="title")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type='messages',
                elem_id="chatbot",
                height=600,
                show_label=False,
                avatar_images=(None, None)
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=4,
                    elem_classes=["input-box"]
                )
                submit_btn = gr.Button("Send 🚀", scale=1)

            with gr.Row():
                new_chat_btn = gr.Button("🔄 New Conversation", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Image Input")
            image_input = gr.Image(
                type="pil",
                label="Upload Image (Optional)",
                elem_id="image-preview"
            )

            with gr.Row():
                clear_img_btn = gr.Button("🗑️ Clear Image", elem_classes=["clear-btn"])

            gr.Markdown("""
            ### 💡 Tips
            - Upload an image to ask visual questions
            - Clear image for text-only chat
            - Conversation context is maintained
            - Use "New Conversation" to reset
            """)

    # Event handlers
    submit_btn.click(
        process_message,
        inputs=[msg_input, image_input, chatbot],
        outputs=[chatbot, msg_input]
    )

    msg_input.submit(
        process_message,
        inputs=[msg_input, image_input, chatbot],
        outputs=[chatbot, msg_input]
    )

    clear_img_btn.click(
        clear_image,
        outputs=[image_input]
    )

    new_chat_btn.click(
        new_conversation,
        outputs=[chatbot, image_input, msg_input]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
