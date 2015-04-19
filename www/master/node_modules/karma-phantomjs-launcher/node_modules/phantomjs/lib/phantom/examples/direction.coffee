# Get driving direction using Google Directions API.

page = require('webpage').create()
system = require 'system'

if system.args.length < 3
  console.log 'Usage: direction.coffee origin destination'
  console.log 'Example: direction.coffee "San Diego" "Palo Alto"'
  phantom.exit 1
else
  origin = system.args[1]
  dest = system.args[2]
  page.open encodeURI('http://maps.googleapis.com/maps/api/directions/xml?origin=' + origin +
                      '&destination=' + dest + '&units=imperial&mode=driving&sensor=false'),
            (status) ->
              if status isnt 'success'
                console.log 'Unable to access network'
              else
                steps = page.content.match(/<html_instructions>(.*)<\/html_instructions>/ig)
                if not steps
                  console.log 'No data available for ' + origin + ' to ' + dest
                else
                  for ins in steps
                    ins = ins.replace(/\&lt;/ig, '<').replace(/\&gt;/ig, '>')
                    ins = ins.replace(/\<div/ig, '\n<div')
                    ins = ins.replace(/<.*?>/g, '')
                    console.log(ins)
                  console.log ''
                  console.log page.content.match(/<copyrights>.*<\/copyrights>/ig).join('').replace(/<.*?>/g, '')
              phantom.exit()
