# Use 'page.injectJs()' to load the script itself in the Page context

if phantom?
  page = require('webpage').create()

  # Route "console.log()" calls from within the Page context to the main
  # Phantom context (i.e. current "this")
  page.onConsoleMessage = (msg) -> console.log(msg)

  page.onAlert = (msg) -> console.log(msg)

  console.log "* Script running in the Phantom context."
  console.log "* Script will 'inject' itself in a page..."
  page.open "about:blank", (status) ->
    if status is "success"
      if page.injectJs("injectme.coffee")
        console.log "... done injecting itself!"
      else
        console.log "... fail! Check the $PWD?!"
    phantom.exit()
else
  alert "* Script running in the Page context."

