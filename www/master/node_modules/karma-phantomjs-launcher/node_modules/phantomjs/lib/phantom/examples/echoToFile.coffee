# echoToFile.coffee - Write in a given file all the parameters passed on the CLI
fs = require 'fs'
system = require 'system'

if system.args.length < 3
  console.log "Usage: echoToFile.coffee DESTINATION_FILE <arguments to echo...>"
  phantom.exit 1
else
  content = ""
  f = null
  i = 2
  while i < system.args.length
    content += system.args[i] + (if i == system.args.length - 1 then "" else " ")
    ++i
  try
    fs.write system.args[1], content, "w"
  catch e
    console.log e
  phantom.exit()
