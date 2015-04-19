# Modify global object at the page initialization.
# In this example, effectively Math.random() always returns 0.42.

page = require('webpage').create()
page.onInitialized = ->
  page.evaluate ->
    Math.random = ->
      42 / 100

page.open "http://ariya.github.com/js/random/", (status) ->
  if status != "success"
    console.log "Network error."
  else
    console.log page.evaluate(->
      document.getElementById("numbers").textContent
    )
  phantom.exit()

