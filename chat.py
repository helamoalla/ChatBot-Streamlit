import streamlit as st
from huggingface_hub import InferenceClient

# Set up the Hugging Face API token securely
api_key = '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'
model_name = 'mistralai/Mistral-7B-Instruct-v0.3'

# Initialize the inference client
client = InferenceClient(model=model_name, token=api_key)

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("What is up?")

# Check if the input is empty and display an alert if necessary
if prompt is not None and prompt.strip() == "":
    st.warning("Vous ne pouvez pas envoyer un message vide. Veuillez entrer un texte.")
else:
    if prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call the Hugging Face model to get the response (improved prompt)
        prompt_input = f"""
        <s>[INST]
        You are an AI assistant from Gnomon Digital that Optimizes decision making via AI-based solutions. Respond to the following user prompt clearly, concisely, and in a deterministic manner.
        Use the same language as the user's input to respond. Keep the response brief (around 2-3 sentences).
        Here is the prompt: {prompt} 
        [/INST]
        """
        result = client.text_generation(
            prompt=prompt_input,
            max_new_tokens=128,
            temperature=0.3,  # Low temperature for more deterministic responses
            top_p=0.95,
            top_k=50
        )
        assistant_response = result  # The full response text from the model

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
