'use strict'

var util = require('util')
  , request = require('../index')


module.exports = function debug() {
  if (request.debug) {
    console.error('REQUEST %s', util.format.apply(util, arguments))
  }
}
