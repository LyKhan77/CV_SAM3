from huggingface_hub import login

print("--- Hugging Face Login Helper ---")
print("This script will help you log in to your Hugging Face account.")
print("Please paste your Hugging Face Access Token below.")
print("You can create a token with 'read' role here: https://huggingface.co/settings/tokens")
print("-" * 35)

# The login function will prompt for a token and save it automatically
login()

print("-" * 35)
print("--- Login Successful! ---")
print("Token has been saved. You can now delete this 'login_helper.py' file.")
print("Please try running the main application again with: python main.py")
