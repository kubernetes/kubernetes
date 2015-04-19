'use strict';

var memoize = require('../..');

module.exports = {
	"": function (t, a) {
		var i = 0, fn = function (x) { ++i; return x; };

		fn = memoize(fn);
		return {
			"No arg": function () {
				i = 0;
				a(fn(), undefined, "First");
				a(fn(), undefined, "Second");
				a(fn(), undefined, "Third");
				a(i, 1, "Called once");
			},
			Arg: function () {
				var x = {};
				i = 0;
				a(fn(x, 8), x, "First");
				a(fn(x, 4), x, "Second");
				a(fn(x, 2), x, "Third");
				a(i, 1, "Called once");
			},
			"Other Arg": function () {
				var x = {};
				i = 0;
				a(fn(x, 2), x, "First");
				a(fn(x, 9), x, "Second");
				a(fn(x, 3), x, "Third");
				a(i, 1, "Called once");
			}
		};
	},
	Delete: function (a) {
		var i = 0, fn, mfn, x = {};

		fn = function (a, b, c) {
			return a + (++i);
		};
		mfn = memoize(fn, { length: 1 });
		a(mfn(3), 4, "Init");
		a(mfn(4, x, 1), 6, "Init #2");
		mfn.delete(4);
		a(mfn(3, x, 1), 4, "Cached");
		mfn(3, x, 1);
		a(i, 2, "Pre clear");
		mfn.delete(3, x, 1);
		a(i, 2, "After clear");
		a(mfn(3, x, 1), 6, "Reinit");
		a(i, 3, "Reinit count");
		a(mfn(3, x, 1), 6, "Reinit Cached");
		a(i, 3, "Reinit count");
	}
};
