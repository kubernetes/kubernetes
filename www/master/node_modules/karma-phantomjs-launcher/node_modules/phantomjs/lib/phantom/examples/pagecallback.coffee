p = require("webpage").create()

p.onConsoleMessage = (msg) ->
  console.log msg

# Calls to "callPhantom" within the page 'p' arrive here
p.onCallback = (msg) ->
  console.log "Received by the 'phantom' main context: " + msg
  "Hello there, I'm coming to you from the 'phantom' context instead"

p.evaluate ->
  # Return-value of the "onCallback" handler arrive here
  callbackResponse = window.callPhantom "Hello, I'm coming to you from the 'page' context"
  console.log "Received by the 'page' context: " + callbackResponse

phantom.exit()
