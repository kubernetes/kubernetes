'use strict';
var path = require('path');

module.exports = exports = function extname (file) {
  var extension = path.extname(file).slice(1);
  var extensionAndHash = extension.split('?');
  return extensionAndHash[0];
};
