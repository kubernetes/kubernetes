page = require('webpage').create()

page.viewportSize = { width: 400, height : 400 }
page.content = '<html><body><canvas id="surface"></canvas></body></html>'

page.evaluate ->
  el = document.getElementById 'surface'
  context = el.getContext '2d'
  width = window.innerWidth
  height = window.innerHeight
  cx = width / 2
  cy = height / 2
  radius = width / 2.3
  i = 0

  el.width = width
  el.height = height
  imageData = context.createImageData(width, height)
  pixels = imageData.data

  for y in [0...height]
    for x in [0...width]
      i = i + 4
      rx = x - cx
      ry = y - cy
      d = rx * rx + ry * ry
      if d < radius * radius
        hue = 6 * (Math.atan2(ry, rx) + Math.PI) / (2 * Math.PI)
        sat = Math.sqrt(d) / radius
        g = Math.floor(hue)
        f = hue - g
        u = 255 * (1 - sat)
        v = 255 * (1 - sat * f)
        w = 255 * (1 - sat * (1 - f))
        pixels[i] = [255, v, u, u, w, 255, 255][g]
        pixels[i + 1] = [w, 255, 255, v, u, u, w][g]
        pixels[i + 2] = [u, u, w, 255, 255, v, u][g]
        pixels[i + 3] = 255

  context.putImageData imageData, 0, 0
  document.body.style.backgroundColor = 'white'
  document.body.style.margin = '0px'

page.render('colorwheel.png')

phantom.exit()
