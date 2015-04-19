fs = require 'fs'
xml2js = require 'xml2js'

parser = new xml2js.Parser

fs.readFile 'canon.xml', (err, data) ->
  console.log err
  parser.parseString (err, result) ->
    console.log err
    console.dir result

