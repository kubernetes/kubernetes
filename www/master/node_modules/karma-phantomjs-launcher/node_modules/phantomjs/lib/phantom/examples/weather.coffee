page = require('webpage').create()
system = require 'system'

city = 'Mountain View, California'; # default
if system.args.length > 1
    city = Array.prototype.slice.call(system.args, 1).join(' ')
url = encodeURI 'http://api.openweathermap.org/data/2.1/find/name?q=' + city

console.log 'Checking weather condition for', city, '...'

page.open url, (status) ->
    if status isnt 'success'
        console.log 'Error: Unable to access network!'
    else
        result = page.evaluate ->
            return document.body.innerText
        try
            data = JSON.parse result
            data = data.list[0]
            console.log ''
            console.log 'City:',  data.name
            console.log 'Condition:', data.weather.map (entry) ->
                return entry.main
            console.log 'Temperature:', Math.round(data.main.temp - 273.15), 'C'
            console.log 'Humidity:', Math.round(data.main.humidity), '%'
        catch e
           console.log 'Error:', e.toString()

    phantom.exit()
