page = require("webpage").create()
system = require("system")

if system.args.length < 2
  console.log "Usage: loadurlwithoutcss.js URL"
  phantom.exit()

address = system.args[1]

page.onResourceRequested = (requestData, request) ->
  if (/http:\/\/.+?\.css/g).test(requestData["url"]) or requestData["Content-Type"] is "text/css"
    console.log "The url of the request is matching. Aborting: " + requestData["url"]
    request.abort()

page.open address, (status) ->
  if status is "success"
    phantom.exit()
  else
    console.log "Unable to load the address!"
    phantom.exit()
