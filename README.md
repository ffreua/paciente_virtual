# Paciente Virtual — v3.5

**Correção importante:** as chaves (GROQ_API_KEY/OPENAI_API_KEY) e o nome do modelo agora ficam em `st.session_state`.  
Isso resolve o aviso "Informe a GROQ_API_KEY na sidebar" mesmo quando a chave já foi digitada.

- Tudo na sidebar (provedor/modelo/chaves, estilo das respostas, prompt).
- Reset limpa histórico + prompt.
- Regras de concisão + truncamento por nº de frases.

## Rodar
```bash
pip install -r requirements.txt
streamlit run app.py
```
