page = require('webpage').create()
system = require 'system'

if system.args.length is 1
  console.log 'Usage: loadspeed.coffee <some URL>'
  phantom.exit 1
else
  t = Date.now()
  address = system.args[1]
  page.open address, (status) ->
    if status isnt 'success'
      console.log('FAIL to load the address')
    else
      t = Date.now() - t
      console.log('Page title is ' + page.evaluate( (-> document.title) ))
      console.log('Loading time ' + t + ' msec')
    phantom.exit()

