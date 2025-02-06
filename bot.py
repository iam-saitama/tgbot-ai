import telebot
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from io import BytesIO

# Токен бота
bot = telebot.TeleBot("7326578244:AAGWWYsRJ3UwFuyWms7pBLAzrLB2QJjD7yY")


# Создание нейронной сети
def create_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Генерация данных
def generate_data(model, input_data):
    input_data = np.array(input_data, dtype=float).reshape(-1, 1)
    predictions = model.predict(input_data)
    return predictions.flatten()

# Создание графика
def create_plot(input_data, output_data):
    plt.figure()
    plt.plot(input_data, output_data, marker='o')
    plt.xlabel('Входные данные')
    plt.ylabel('Сгенерированные данные')
    plt.title('График, созданный нейронной сетью')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне набор чисел через запятую, и я создам график на основе нейронной сети.")

# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        # Получаем данные от пользователя
        user_input = message.text
        input_data = list(map(float, user_input.split(',')))

        # Создание и обучение модели
        model = create_model()
        output_data = generate_data(model, input_data)

        # Создание графика
        plot_buffer = create_plot(input_data, output_data)

        # Отправка графика пользователю
        bot.send_photo(message.chat.id, plot_buffer)
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {e}")

# Запуск бота
bot.polling()