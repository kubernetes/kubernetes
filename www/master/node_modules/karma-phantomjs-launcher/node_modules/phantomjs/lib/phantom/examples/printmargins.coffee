page = require("webpage").create()
system = require("system")
if system.args.length < 7
  console.log "Usage: printmargins.js URL filename LEFT TOP RIGHT BOTTOM"
  console.log "  margin examples: \"1cm\", \"10px\", \"7mm\", \"5in\""
  phantom.exit 1
else
  address = system.args[1]
  output = system.args[2]
  marginLeft = system.args[3]
  marginTop = system.args[4]
  marginRight = system.args[5]
  marginBottom = system.args[6]
  page.viewportSize =
    width: 600
    height: 600

  page.paperSize =
    format: "A4"
    margin:
      left: marginLeft
      top: marginTop
      right: marginRight
      bottom: marginBottom

  page.open address, (status) ->
    if status isnt "success"
      console.log "Unable to load the address!"
    else
      window.setTimeout (->
        page.render output
        phantom.exit()
      ), 200
