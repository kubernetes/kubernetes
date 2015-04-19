# List all the files in a Tree of Directories
system = require 'system'

if system.args.length != 2
  console.log "Usage: phantomjs scandir.coffee DIRECTORY_TO_SCAN"
  phantom.exit 1
scanDirectory = (path) ->
  fs = require 'fs'
  if fs.exists(path) and fs.isFile(path)
    console.log path
  else if fs.isDirectory(path)
    fs.list(path).forEach (e) ->
      scanDirectory path + "/" + e  if e != "." and e != ".."

scanDirectory system.args[1]
phantom.exit()
