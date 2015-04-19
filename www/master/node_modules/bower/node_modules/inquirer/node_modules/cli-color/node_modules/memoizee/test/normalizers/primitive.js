'use strict';

var memoize = require('../..')

  , join = Array.prototype.join;

module.exports = function (a) {
	var i = 0, fn = function () { ++i; return join.call(arguments, '|'); }
	  , y = { toString: function () { return 'foo'; } }, mfn;
	mfn = memoize(fn, { primitive: true, length: false });
	a(mfn(y, 'bar', 'zeta'), 'foo|bar|zeta', "#1");
	a(mfn('foo', 'bar', 'zeta'), 'foo|bar|zeta', "#2");
	a(i, 1, "Called once");
	a(mfn(y, 'bar'), 'foo|bar', "#3");
	a(i, 2, "Called twice");
	a(mfn(y, 'bar'), 'foo|bar', "#4");
	a(i, 2, "Called twice #2");
};
