// phantomjs test script
// opens url and reports time to load
// requires an active internet connection
var page = require('webpage').create()
var system = require('system')
var t
var address

if (system.args.length === 1) {
  console.log('Usage: loadspeed.js <some URL>')
  phantom.exit()
}

t = Date.now()
address = system.args[1]
page.open(address, function (status) {
  if (status !== 'success') {
    console.log('FAIL to load the address')
  } else {
    t = Date.now() - t
    console.log('Loading time ' + t + ' msec')
  }

  phantom.exit()
})