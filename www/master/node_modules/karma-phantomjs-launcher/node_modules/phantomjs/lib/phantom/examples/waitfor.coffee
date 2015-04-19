##
# Wait until the test condition is true or a timeout occurs. Useful for waiting
# on a server response or for a ui change (fadeIn, etc.) to occur.
#
# @param testFx javascript condition that evaluates to a boolean,
# it can be passed in as a string (e.g.: "1 == 1" or "$('#bar').is(':visible')" or
# as a callback function.
# @param onReady what to do when testFx condition is fulfilled,
# it can be passed in as a string (e.g.: "1 == 1" or "$('#bar').is(':visible')" or
# as a callback function.
# @param timeOutMillis the max amount of time to wait. If not specified, 3 sec is used.
##
waitFor = (testFx, onReady, timeOutMillis=3000) ->
  start = new Date().getTime()
  condition = false
  f = ->
    if (new Date().getTime() - start < timeOutMillis) and not condition
      # If not time-out yet and condition not yet fulfilled
      condition = (if typeof testFx is 'string' then eval testFx else testFx()) #< defensive code
    else
      if not condition
        # If condition still not fulfilled (timeout but condition is 'false')
        console.log "'waitFor()' timeout"
        phantom.exit 1
      else
        # Condition fulfilled (timeout and/or condition is 'true')
        console.log "'waitFor()' finished in #{new Date().getTime() - start}ms."
        if typeof onReady is 'string' then eval onReady else onReady() #< Do what it's supposed to do once the condition is fulfilled
        clearInterval interval #< Stop this interval
  interval = setInterval f, 250 #< repeat check every 250ms


page = require('webpage').create()

# Open Twitter on 'sencha' profile and, onPageLoad, do...
page.open 'http://twitter.com/#!/sencha', (status) ->
  # Check for page load success
  if status isnt 'success'
    console.log 'Unable to access network'
  else
    # Wait for 'signin-dropdown' to be visible
    waitFor ->
      # Check in the page if a specific element is now visible
      page.evaluate ->
        $('#signin-dropdown').is ':visible'
    , ->
       console.log 'The sign-in dialog should be visible now.'
       phantom.exit()
