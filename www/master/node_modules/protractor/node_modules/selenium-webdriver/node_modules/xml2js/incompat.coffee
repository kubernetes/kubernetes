{parseString} = require './lib/xml2js'
xml = '<outline htmlUrl="futurity.org" text="Futurity.org" title="Futurity.org" type="rss" xmlUrl="http://www.futurity.org/feed/"/>'
parseString xml, (err, result) ->
    console.dir result

