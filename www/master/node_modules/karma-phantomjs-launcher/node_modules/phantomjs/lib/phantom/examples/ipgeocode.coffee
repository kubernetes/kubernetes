# Give the estimated location based on the IP address.

window.cb = (data) ->
  loc = data.city
  if data.region_name.length > 0
    loc = loc + ', ' + data.region_name
  console.log 'IP address: ' + data.ip
  console.log 'Estimated location: ' + loc
  phantom.exit()

el = document.createElement 'script'
el.src = 'http://freegeoip.net/json/?callback=window.cb'
document.body.appendChild el
