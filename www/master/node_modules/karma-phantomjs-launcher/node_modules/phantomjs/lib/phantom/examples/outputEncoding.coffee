helloWorld = () -> console.log phantom.outputEncoding + ": こんにちは、世界！"

console.log "Using default encoding..."
helloWorld()

console.log "\nUsing other encodings..."
for enc in ["euc-jp", "sjis", "utf8", "System"]
  do (enc) ->
    phantom.outputEncoding = enc
    helloWorld()

phantom.exit()
