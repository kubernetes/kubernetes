page = require('webpage').create()

console.log 'The default user agent is ' + page.settings.userAgent

page.settings.userAgent = 'SpecialAgent'
page.open 'http://www.httpuseragent.org', (status) ->
  if status isnt 'success'
    console.log 'Unable to access network'
  else
    console.log page.evaluate -> document.getElementById('myagent').innerText
  phantom.exit()
