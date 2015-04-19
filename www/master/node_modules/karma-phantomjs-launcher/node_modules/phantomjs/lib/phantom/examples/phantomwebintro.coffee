# Read the Phantom webpage '#intro' element text using jQuery and "includeJs"

page = require('webpage').create()

page.onConsoleMessage = (msg) -> console.log msg

page.open "http://www.phantomjs.org", (status) ->
  if status is "success"
    page.includeJs "http://ajax.googleapis.com/ajax/libs/jquery/1.6.1/jquery.min.js", ->
      page.evaluate ->
        console.log "$(\"#intro\").text() -> " + $("#intro").text()
      phantom.exit()

