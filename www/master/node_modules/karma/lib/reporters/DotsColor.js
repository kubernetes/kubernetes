var DotsReporter = require('./Dots');
var BaseColorReporter = require('./BaseColor');


var DotsColorReporter = function(formatError, reportSlow) {
  DotsReporter.call(this, formatError, reportSlow);
  BaseColorReporter.call(this);
};


// PUBLISH
module.exports = DotsColorReporter;
