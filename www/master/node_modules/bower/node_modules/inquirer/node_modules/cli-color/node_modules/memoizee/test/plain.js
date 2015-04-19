'use strict';

module.exports = function (t) {
	return {
		"": function (a) {
			var i = 0, fn = function (x) { ++i; return x; }, mfn
			  , y = { toString: function () { return 'foo'; } };
			mfn = t(fn, { primitive: true });
			a(typeof mfn, 'function', "Returns");
			a(mfn.__memoized__, true, "Marked");
			a(t(mfn), mfn, "Do not memoize memoized");
			a(mfn(y), y, "#1");
			a(mfn('foo'), y, "#2");
			a(i, 1, "Called once");
		},
		"Clear cache": function (a) {
			var i = 0, fn = function (x, y, z) { ++i; return x + y + z; }, mfn
			  , y = { toString: function () { return 'foo'; } };
			mfn = t(fn, { primitive: true });
			a(mfn(y, 'bar', 'zeta'), 'foobarzeta', "#1");
			a(mfn('foo', 'bar', 'zeta'), 'foobarzeta', "#2");
			a(i, 1, "Called once");
			mfn.delete('foo', { toString: function () { return 'bar'; } },
				'zeta');
			a(mfn(y, 'bar', 'zeta'), 'foobarzeta', "#3");
			a(i, 2, "Called twice");
		}
	};
};
