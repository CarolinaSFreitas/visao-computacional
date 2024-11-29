# Projeto de Visão Computacional
Projeto criado para disciplina Fundamentos de Inteligência Artificial (FIA) - Graduação. Prof. Pablo De Chiaro

Este projeto configura um ambiente virtual Python e instala as bibliotecas necessárias para um projeto de Visão Computacional.


### Passos para criar e ativar um ambiente virtual:

1. **Criar o ambiente virtual:**

   ```bash
   python -m venv env-visao
   ```

2. **Ativar o ambiente virtual:**

   No macOS e Linux:

   ```bash
   source ./env-visao/bin/activate
   ```

   No Windows:

   ```bash
   .\env-visao\Scripts\activate
   ```
## Instalação de Dependências

Certifique-se de que seu ambiente virtual esteja ativado. Instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Conteúdo do arquivo `requirements.txt`:

```text
numpy==2.0.0
opencv-python==4.10.0.84
```

# Instalar Tensorflow
``pip install tensorflow``

# Rodar o Script
``python deteccao_gato.py``
