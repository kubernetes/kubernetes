'use strict';
var uglfiy = require('uglify-js');
var minifier = require('./minifier');

module.exports = function(opts) {
  return minifier(opts, uglfiy);
};
