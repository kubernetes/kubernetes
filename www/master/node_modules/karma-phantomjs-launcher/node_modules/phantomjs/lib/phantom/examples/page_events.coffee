# The purpose of this is to show how and when events fire, considering 5 steps
# happening as follows:
#
#      1. Load URL
#      2. Load same URL, but adding an internal FRAGMENT to it
#      3. Click on an internal Link, that points to another internal FRAGMENT
#      4. Click on an external Link, that will send the page somewhere else
#      5. Close page
#
# Take particular care when going through the output, to understand when
# things happen (and in which order). Particularly, notice what DOESN'T
# happen during step 3.
#
# If invoked with "-v" it will print out the Page Resources as they are
# Requested and Received.
#
# NOTE.1: The "onConsoleMessage/onAlert/onPrompt/onConfirm" events are
# registered but not used here. This is left for you to have fun with.
# NOTE.2: This script is not here to teach you ANY JavaScript. It's aweful!
# NOTE.3: Main audience for this are people new to PhantomJS.
printArgs = ->
  i = undefined
  ilen = undefined
  i = 0
  ilen = arguments_.length

  while i < ilen
    console.log "    arguments[" + i + "] = " + JSON.stringify(arguments_[i])
    ++i
  console.log ""
sys = require("system")
page = require("webpage").create()
logResources = false
step1url = "http://en.wikipedia.org/wiki/DOM_events"
step2url = "http://en.wikipedia.org/wiki/DOM_events#Event_flow"
logResources = true  if sys.args.length > 1 and sys.args[1] is "-v"

#//////////////////////////////////////////////////////////////////////////////
page.onInitialized = ->
  console.log "page.onInitialized"
  printArgs.apply this, arguments_

page.onLoadStarted = ->
  console.log "page.onLoadStarted"
  printArgs.apply this, arguments_

page.onLoadFinished = ->
  console.log "page.onLoadFinished"
  printArgs.apply this, arguments_

page.onUrlChanged = ->
  console.log "page.onUrlChanged"
  printArgs.apply this, arguments_

page.onNavigationRequested = ->
  console.log "page.onNavigationRequested"
  printArgs.apply this, arguments_

if logResources is true
  page.onResourceRequested = ->
    console.log "page.onResourceRequested"
    printArgs.apply this, arguments_

  page.onResourceReceived = ->
    console.log "page.onResourceReceived"
    printArgs.apply this, arguments_
page.onClosing = ->
  console.log "page.onClosing"
  printArgs.apply this, arguments_


# window.console.log(msg);
page.onConsoleMessage = ->
  console.log "page.onConsoleMessage"
  printArgs.apply this, arguments_


# window.alert(msg);
page.onAlert = ->
  console.log "page.onAlert"
  printArgs.apply this, arguments_


# var confirmed = window.confirm(msg);
page.onConfirm = ->
  console.log "page.onConfirm"
  printArgs.apply this, arguments_


# var user_value = window.prompt(msg, default_value);
page.onPrompt = ->
  console.log "page.onPrompt"
  printArgs.apply this, arguments_


#//////////////////////////////////////////////////////////////////////////////
setTimeout (->
  console.log ""
  console.log "### STEP 1: Load '" + step1url + "'"
  page.open step1url
), 0
setTimeout (->
  console.log ""
  console.log "### STEP 2: Load '" + step2url + "' (load same URL plus FRAGMENT)"
  page.open step2url
), 5000
setTimeout (->
  console.log ""
  console.log "### STEP 3: Click on page internal link (aka FRAGMENT)"
  page.evaluate ->
    ev = document.createEvent("MouseEvents")
    ev.initEvent "click", true, true
    document.querySelector("a[href='#Event_object']").dispatchEvent ev

), 10000
setTimeout (->
  console.log ""
  console.log "### STEP 4: Click on page external link"
  page.evaluate ->
    ev = document.createEvent("MouseEvents")
    ev.initEvent "click", true, true
    document.querySelector("a[title='JavaScript']").dispatchEvent ev

), 15000
setTimeout (->
  console.log ""
  console.log "### STEP 5: Close page and shutdown (with a delay)"
  page.close()
  setTimeout (->
    phantom.exit()
  ), 100
), 20000