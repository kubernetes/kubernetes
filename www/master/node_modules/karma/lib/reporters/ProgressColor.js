var ProgressReporter = require('./Progress');
var BaseColorReporter = require('./BaseColor');


var ProgressColorReporter = function(formatError, reportSlow) {
  ProgressReporter.call(this, formatError, reportSlow);
  BaseColorReporter.call(this);
};


// PUBLISH
module.exports = ProgressColorReporter;
