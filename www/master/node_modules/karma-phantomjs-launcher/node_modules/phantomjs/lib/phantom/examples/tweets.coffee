# Get twitter status for given account (or for the default one, "PhantomJS")

page = require('webpage').create()
system = require 'system'
twitterId = 'PhantomJS' #< default value

# Route "console.log()" calls from within the Page context to the main Phantom context (i.e. current "this")
page.onConsoleMessage = (msg) ->
  console.log msg

# Print usage message, if no twitter ID is passed
if system.args.length < 2
  console.log 'Usage: tweets.coffee [twitter ID]'
else
  twitterId = system.args[1]

# Heading
console.log "*** Latest tweets from @#{twitterId} ***\n"

# Open Twitter Mobile and, onPageLoad, do...
page.open encodeURI("http://mobile.twitter.com/#{twitterId}"), (status) ->
  # Check for page load success
  if status isnt 'success'
    console.log 'Unable to access network'
  else
    # Execute some DOM inspection within the page context
    page.evaluate ->
      list = document.querySelectorAll 'div.tweet-text'
      for i, j in list
        console.log "#{j + 1}: #{i.innerText}"
  phantom.exit()
