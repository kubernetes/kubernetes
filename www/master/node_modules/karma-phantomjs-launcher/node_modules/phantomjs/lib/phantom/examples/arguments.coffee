system = require 'system'
if system.args.length is 1
  console.log 'Try to pass some args when invoking this script!'
else
  for arg, i in system.args
    console.log i + ': ' + arg
phantom.exit()
