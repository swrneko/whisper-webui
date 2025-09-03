import gradio as gr
import whisper
from pathlib import Path

OUTPUT_PATH='outputs'

# Список моделей wisper
wisperModels = ['tiny', 'base', 'small', 'medium', 'large', 'turbo'] 

# Функция распознования текста
def recognize(model, audioFile):
    model = whisper.load_model(model)
    outputText = model.transcribe(audioFile)
    return outputText['text']

def saveFile(filename, text):
    directory = Path(OUTPUT_PATH)
    filePath = directory / filename
    filePath.parent.mkdir(parents=True, exist_ok=True)
    filePath.write_text(text, encoding='utf-8')

# Запуск 
with gr.Blocks() as demo:
    gr.HTML('''
    <div align='center'>
        <h1>
            OpenAI Whisper WebUI by <a href='https://github.com/swrneko/whisper-webui'>swrneko</>
        </h1>
    </div>
    ''')
    # Создаю ряд
    with gr.Row():
        # Создаю колонку
        with gr.Column():
            # Загрузка аудио
            audioFile = gr.Audio(type="filepath", label='Load audio for text recognize')
            # Сохранение распознанного текста в файл
            filename = gr.Textbox(label='Output filename', value='output.txt', interactive=True, placeholder='output.txt')

        # Создаю колонку
        with gr.Column():
            # Дропдаун для выбора модели
            selectedModel = gr.Dropdown(wisperModels, interactive=True, label='Select model')
            # Кнопка распознавания текста
            recognizeBtn = gr.Button("Recognize", variant='primary')
            # Выходной текст
            outputText = gr.TextArea(label='Recognized text')

    recognizeBtn.click(recognize, inputs=[selectedModel, audioFile], outputs=[outputText])
    outputText.change(saveFile, inputs=[filename, outputText])

demo.launch()
