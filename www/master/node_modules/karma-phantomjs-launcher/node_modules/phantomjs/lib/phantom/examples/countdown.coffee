t = 10
interval = setInterval ->
  if t > 0
    console.log t--
  else
    console.log 'BLAST OFF!'
    phantom.exit()
, 1000
