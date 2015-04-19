###
Sort integers from the command line in a very ridiculous way: leveraging timeouts :P
###

system = require 'system'

if system.args.length < 2
  console.log "Usage: phantomjs sleepsort.coffee PUT YOUR INTEGERS HERE SEPARATED BY SPACES"
  phantom.exit 1
else
  sortedCount = 0
  args = Array.prototype.slice.call(system.args, 1)
  for int in args
    setTimeout (do (int) ->
      ->
        console.log int
        ++sortedCount
        phantom.exit() if sortedCount is args.length),
      int

