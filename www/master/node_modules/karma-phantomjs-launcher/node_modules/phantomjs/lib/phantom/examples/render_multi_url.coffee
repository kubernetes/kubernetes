# Render Multiple URLs to file

system = require("system")

# Render given urls
# @param array of URLs to render
# @param callbackPerUrl Function called after finishing each URL, including the last URL
# @param callbackFinal Function called after finishing everything
RenderUrlsToFile = (urls, callbackPerUrl, callbackFinal) ->
  urlIndex = 0 # only for easy file naming
  webpage = require("webpage")
  page = null
  getFilename = ->
    "rendermulti-" + urlIndex + ".png"

  next = (status, url, file) ->
    page.close()
    callbackPerUrl status, url, file
    retrieve()

  retrieve = ->
    if urls.length > 0
      url = urls.shift()
      urlIndex++
      page = webpage.create()
      page.viewportSize =
        width: 800
        height: 600

      page.settings.userAgent = "Phantom.js bot"
      page.open "http://" + url, (status) ->
        file = getFilename()
        if status is "success"
          window.setTimeout (->
            page.render file
            next status, url, file
          ), 200
        else
          next status, url, file

    else
      callbackFinal()

  retrieve()
arrayOfUrls = null
if system.args.length > 1
  arrayOfUrls = Array::slice.call(system.args, 1)
else
  # Default (no args passed)
  console.log "Usage: phantomjs render_multi_url.js [domain.name1, domain.name2, ...]"
  arrayOfUrls = ["www.google.com", "www.bbc.co.uk", "www.phantomjs.org"]

RenderUrlsToFile arrayOfUrls, ((status, url, file) ->
  if status isnt "success"
    console.log "Unable to render '" + url + "'"
  else
    console.log "Rendered '" + url + "' at '" + file + "'"
), ->
  phantom.exit()

