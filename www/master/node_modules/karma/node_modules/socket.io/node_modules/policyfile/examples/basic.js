var http = require('http')
  , fspfs = require('../');

var flash = fspfs.createServer();
flash.listen();