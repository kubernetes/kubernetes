page = require('webpage').create()
system = require 'system'

if system.args.length is 1
  console.log 'Usage: netlog.coffee <some URL>'
  phantom.exit 1
else
  address = system.args[1]
  page.onResourceRequested = (req) ->
    console.log 'requested ' + JSON.stringify(req, undefined, 4)

  page.onResourceReceived = (res) ->
    console.log 'received ' + JSON.stringify(res, undefined, 4)

  page.open address, (status) ->
    if status isnt 'success'
      console.log 'FAIL to load the address'
    phantom.exit()
