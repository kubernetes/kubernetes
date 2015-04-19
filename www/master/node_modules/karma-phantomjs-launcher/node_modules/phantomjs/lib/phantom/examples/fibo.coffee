fibs = [0, 1]
f = ->
  console.log fibs[fibs.length - 1]
  fibs.push fibs[fibs.length - 1] + fibs[fibs.length - 2]
  if fibs.length > 10
    window.clearInterval ticker
    phantom.exit()
ticker = window.setInterval(f, 300)
