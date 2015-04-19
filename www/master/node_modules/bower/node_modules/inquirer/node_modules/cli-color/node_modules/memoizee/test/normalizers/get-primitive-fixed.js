'use strict';

var memoize = require('../..');

module.exports = {
	"": function (a) {
		var i = 0, fn = function (x, y, z) { ++i; return x + y + z; }, mfn
		  , y = { toString: function () { return 'foo'; } };
		mfn = memoize(fn, { primitive: true });
		a(mfn(y, 'bar', 'zeta'), 'foobarzeta', "#1");
		a(mfn('foo', 'bar', 'zeta'), 'foobarzeta', "#2");
		a(i, 1, "Called once");
	},
	Delete: function (a) {
		var i = 0, fn = function (x, y, z) { ++i; return x + y + z; }, mfn
		  , y = { toString: function () { return 'foo'; } };
		mfn = memoize(fn, { primitive: true });
		a(mfn(y, 'bar', 'zeta'), 'foobarzeta', "#1");
		a(mfn('foo', 'bar', 'zeta'), 'foobarzeta', "#2");
		a(i, 1, "Called once");
		mfn.delete('foo', { toString: function () { return 'bar'; } },
			'zeta');
		a(mfn(y, 'bar', 'zeta'), 'foobarzeta', "#3");
		a(i, 2, "Called twice");
	},
	Clear: function (a) {
		var i = 0, fn;
		fn = memoize(function (x) {
			if (++i < 2) fn(x);
		});
		a.throws(function () {
			fn('foo');
		}, 'CIRCULAR_INVOCATION');

		i = 0;
		fn = memoize(function (x, y) {
			if (++i < 2) fn(x, y);
		});
		a.throws(function () {
			fn('foo', 'bar');
		}, 'CIRCULAR_INVOCATION');
	}
};
