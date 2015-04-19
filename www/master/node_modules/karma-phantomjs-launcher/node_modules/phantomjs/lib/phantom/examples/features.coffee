feature = undefined
supported = []
unsupported = []
phantom.injectJs "modernizr.js"
console.log "Detected features (using Modernizr " + Modernizr._version + "):"
for feature of Modernizr
  if Modernizr.hasOwnProperty(feature)
    if feature[0] isnt "_" and typeof Modernizr[feature] isnt "function" and feature isnt "input" and feature isnt "inputtypes"
      if Modernizr[feature]
        supported.push feature
      else
        unsupported.push feature
console.log ""
console.log "Supported:"
supported.forEach (e) ->
  console.log "  " + e

console.log ""
console.log "Not supported:"
unsupported.forEach (e) ->
  console.log "  " + e

phantom.exit()