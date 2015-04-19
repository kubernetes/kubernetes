page = require('webpage').create()

page.viewportSize = { width: 320, height: 480 }

page.open 'http://news.google.com/news/i/section?&topic=t',
  (status) ->
    if status isnt 'success'
      console.log 'Unable to access the network!'
    else
      page.evaluate ->
        body = document.body
        body.style.backgroundColor = '#fff'
        body.querySelector('div#title-block').style.display = 'none'
        body.querySelector('form#edition-picker-form')
          .parentElement.parentElement.style.display = 'none'
      page.render 'technews.png'
    phantom.exit()
