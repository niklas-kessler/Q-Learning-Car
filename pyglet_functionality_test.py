import pyglet as pg
from pyglet.window import key, mouse

pg.resource.path = ['./resources']

window = pg.window.Window()
label = pg.text.Label('Hello world!', x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center',
                          font_name='Times New Roman', font_size=30,
                          color=(180,180,0,255))
line = pg.shapes.Line(25,100,100,25)
image = pg.resource.image('car.png')


@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.A:
        print('"A" was pressed.')
    else:
        print('Something else was pressed: ', symbol)


@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        print("The left mouse button was pressed at position ({},{}).".format(x, y))


@window.event
def on_draw():
    window.clear()
    label.draw()
    image.blit(window.width//1.5, window.height//1.5)
    line.draw()


pg.app.run()
