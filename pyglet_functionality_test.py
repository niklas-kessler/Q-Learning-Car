import pyglet
from pyglet.window import key, mouse

pyglet.resource.path = ['./resources']

window = pyglet.window.Window()
label = pyglet.text.Label('Hello world!', x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center',
                          font_name='Times New Roman', font_size=30,
                          color=(120,120,0,255))
image = pyglet.resource.image('car.png')


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

pyglet.app.run()
