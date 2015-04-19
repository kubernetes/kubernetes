'use strict';
module.exports = function (fn, errMsg) {
	if (typeof fn !== 'function') {
		throw new TypeError('Expected a function.');
	}

	var ret;
	var called = false;
	var fnName = fn.name || (/function ([^\(]+)/.exec(fn.toString()) || [])[1];

	return function () {
		if (called) {
			if (errMsg === true) {
				fnName = fnName ? fnName + '()' : 'Function';
				throw new Error(fnName + ' can only be called once.');
			}
			return ret;
		}
		called = true;
		ret = fn.apply(this, arguments);
		fn = null;
		return ret;
	};
};
