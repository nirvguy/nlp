import os
import logging
import configparser
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import surnames
from surnames.predict import Predictor

def message_handler(*args, **kwargs):
    def closure(f):
        f.handler = ( args, kwargs )
        return f
    return closure

def log(f):
    def closure(self, message):
        print("[{message.chat.last_name}, {message.chat.first_name}, {message.chat.id}] {message.text}".format(message=message))
        return f(self, message)
    return closure

logging.basicConfig(level=logging.INFO)
WEIGHTS_PATH = os.path.join('output', 'surnames', 'weights', '0.pth')

class NLPBot(object):
    def __init__(self, token):
        self.bot = Bot(token=token)
        self.dp = Dispatcher(self.bot)
        self.surname_predictor = Predictor(weights_path=WEIGHTS_PATH, k=3)
        self._commands = []
        for member in dir(self):
            member = getattr(self, member)
            if hasattr(member, 'handler'):
                args, kwargs = member.handler

                commands = kwargs.get('commands')
                if commands:
                    self._commands.extend(map(lambda x: '/' + x, commands))

                self.dp.message_handler(*args, **kwargs)(member)

    @message_handler(commands=['start'])
    @log
    async def start(self, message: types.Message):
        await message.reply("Hi!\n"
                            "My commands are: {}".format(", ".join(self._commands)))

    @message_handler(commands=['surname'])
    @log
    async def surname(self, message: types.Message):
        args= message.text.split(' ')
        if len(args) < 2:
            return
        name = message.text.split(' ')[1]
        result = 'Surname origin of {}: \n'.format(name)
        output = self.surname_predictor(name)
        for origin, proba in output:
            result += '{}: {:.3f}%\n'.format(origin, proba)
        print("Answer: ", result)
        await message.reply(result)

    def start_polling(self):
        executor.start_polling(self.dp)

def main():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini') 
    print(config_path)
    config.read(config_path)
    bot = NLPBot(token=config['BOT']['APIKEY'])
    bot.start_polling()

if __name__ == '__main__':
    main()
