import gradio as gr

def responder(pergunta, historico):
    # TODO: chamar seu agente RAG aqui.
    return historico + [(pergunta, "✅ Deploy OK! Agora plugue seu agente RAG aqui.")], ""

with gr.Blocks(title="Agente Brisanet") as demo:
    gr.Markdown("## Agente Brisanet — Fly.io via GitHub Actions")
    chat = gr.Chatbot(height=380)
    txt = gr.Textbox(placeholder="Digite sua pergunta...")
    txt.submit(responder, [txt, chat], [chat, txt])

demo.launch(server_name="0.0.0.0", server_port=8080)
