xml2js = require 'xml2js'
util = require 'util'

body = '<sample><chartest desc="Test for CHARs">Character data here!</chartest></sample>'
xml2js.parseString body, (err, result) ->
  console.log util.inspect result, false, null
