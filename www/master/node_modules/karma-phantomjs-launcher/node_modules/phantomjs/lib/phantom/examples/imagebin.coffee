# Upload an image to imagebin.org

page = require('webpage').create()
system = require 'system'

if system.args.length isnt 2
  console.log 'Usage: imagebin.coffee filename'
  phantom.exit 1
else
  fname = system.args[1]
  page.open 'http://imagebin.org/index.php?page=add', ->
    page.uploadFile 'input[name=image]', fname
    page.evaluate ->
      document.querySelector('input[name=nickname]').value = 'phantom'
      document.querySelector('input[name=disclaimer_agree]').click()
      document.querySelector('form').submit()

    window.setTimeout ->
      phantom.exit()
    , 3000
