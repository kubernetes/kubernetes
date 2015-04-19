port = undefined
server = undefined
service = undefined
system = require("system")
if system.args.length isnt 2
  console.log "Usage: serverkeepalive.js <portnumber>"
  phantom.exit 1
else
  port = system.args[1]
  server = require("webserver").create()
  service = server.listen(port,
    keepAlive: true
  , (request, response) ->
    console.log "Request at " + new Date()
    console.log JSON.stringify(request, null, 4)
    body = JSON.stringify(request, null, 4)
    response.statusCode = 200
    response.headers =
      Cache: "no-cache"
      "Content-Type": "text/plain"
      Connection: "Keep-Alive"
      "Keep-Alive": "timeout=5, max=100"
      "Content-Length": body.length

    response.write body
    response.close()
  )
  if service
    console.log "Web server running on port " + port
  else
    console.log "Error: Could not create web server listening on port " + port
    phantom.exit()