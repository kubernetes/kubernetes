page = require('webpage').create()
system = require 'system'

if system.args.length < 3 or system.args.length > 4
  console.log 'Usage: rasterize.coffee URL filename [paperwidth*paperheight|paperformat]'
  console.log '  paper (pdf output) examples: "5in*7.5in", "10cm*20cm", "A4", "Letter"'
  phantom.exit 1
else
  address = system.args[1]
  output = system.args[2]
  page.viewportSize = { width: 600, height: 600 }
  if system.args.length is 4 and system.args[2].substr(-4) is ".pdf"
    size = system.args[3].split '*'
    if size.length is 2
      page.paperSize = { width: size[0], height: size[1], border: '0px' }
    else
      page.paperSize = { format: system.args[3], orientation: 'portrait', border: '1cm' }
  page.open address, (status) ->
    if status isnt 'success'
      console.log 'Unable to load the address!'
      phantom.exit()
    else
      window.setTimeout (-> page.render output; phantom.exit()), 200
