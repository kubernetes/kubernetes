'use strict';

var memoize = require('../plain');

module.exports = function (t, a) {
	memoize(function () {})();
	a(typeof t.statistics, 'object', "Access to statistics");
	a(Object.keys(t.statistics).length > 0, true, "Statistics collected");
	a(typeof t.log, 'function', "Access to log function");
	a(typeof t.log(), 'string', "Log outputs string");
};
