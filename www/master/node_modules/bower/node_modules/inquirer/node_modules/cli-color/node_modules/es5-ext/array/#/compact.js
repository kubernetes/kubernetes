// Inspired by: http://documentcloud.github.com/underscore/#compact

'use strict';

var filter = Array.prototype.filter;

module.exports = function () {
	return filter.call(this, function (val) { return val != null; });
};
