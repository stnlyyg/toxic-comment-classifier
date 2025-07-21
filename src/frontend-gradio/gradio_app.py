import gradio as gr
import requests

CLASSIFY_COMMENT_API = "http://backend:90/comment_classifier/"

def classify_comments(comment: str):
    if not comment or not comment.strip():
        return {"Error": "Comment can't be blank or whitespace"}
    
    payload = {"comment": comment}
    response = requests.post(CLASSIFY_COMMENT_API, json=payload)
    response.raise_for_status() 
    result = response.json().get("result", "Classification failed.")
    return result

classifier_app = gr.Interface(
    fn=classify_comments,
    inputs=gr.TextArea(placeholder="Enter a comment here...", label="Input comment"),
    outputs=gr.TextArea(label="Comment classified as: "),
    title="Toxic comment classifier"
)

classifier_app.launch(server_name="0.0.0.0")