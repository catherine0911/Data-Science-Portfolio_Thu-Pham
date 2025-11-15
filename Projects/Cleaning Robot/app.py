from Map import Map
from Bot import Bot
from Game import Game
import importlib
import traceback



settings = {
'mapName' : "map1.csv",
'nrCols' : 30,
'nrRows' : 30,
'nrStains' : 10,
'nrPillars' : 0,
'nrWalls' : 0,
'sizeStains' : 2,
'sizePillars' : 0,
'sizeWalls' : 0,
'checkpoint' : [1,1],
}


MAX_STEPS = 2*settings['nrRows']*settings['nrCols']
LATENCY = 0.1
VISUALS = True
CLS = True

botName = 'Bot_2131850'
module = importlib.import_module(botName)
cls = getattr(module, botName)

myMap = Map(settings)
bot = cls(settings)
if not getattr(bot, 'name', False):
	bot.setName(botName)
game = Game(bot, myMap, MAX_STEPS, LATENCY, VISUALS, CLS)


try:
	game.play()
except Exception:
	print("Encountered error: ", traceback.format_exc())