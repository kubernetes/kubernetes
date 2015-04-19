# List movies from kids-in-mind.com

window.cbfunc = (data) ->
  globaldata = data
  list = data.query.results.movie
  for item in list
    console.log item.title + ' [' + item.rating.MPAA.content + ']'
  phantom.exit()

el = document.createElement 'script'
el.src =
"http://query.yahooapis.com/v1/public/yql?q=select%20*%20from%20movies.kids-in-mind&format=json&env=store%3A%2F%2Fdatatables.org%2Falltableswithkeys&callback=window.cbfunc"
document.body.appendChild el
