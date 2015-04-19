'use strict';

var memoize = require('../..');

module.exports = {
	"": function (a) {
		var i = 0, fn = function (x, y, z) { ++i; return [x, y, z]; }, r;

		fn = memoize(fn);
		return {
			"No args": function () {
				i = 0;
				a.deep(r = fn(), [undefined, undefined, undefined], "First");
				a(fn(), r, "Second");
				a(fn(), r, "Third");
				a(i, 1, "Called once");
			},
			"Some Args": function () {
				var x = {};
				i = 0;
				a.deep(r = fn(x, 8), [x, 8, undefined], "First");
				a(fn(x, 8), r, "Second");
				a(fn(x, 8), r, "Third");
				a(i, 1, "Called once");
				return {
					Other: function () {
						a.deep(r = fn(x, 5), [x, 5, undefined], "Second");
						a(fn(x, 5), r, "Third");
						a(i, 2, "Called once");
					}
				};
			},
			"Full stuff": function () {
				var x = {};
				i = 0;
				a.deep(r = fn(x, 8, 23, 98), [x, 8, 23], "First");
				a(fn(x, 8, 23, 43), r, "Second");
				a(fn(x, 8, 23, 9), r, "Third");
				a(i, 1, "Called once");
				return {
					Other: function () {
						a.deep(r = fn(x, 23, 8, 13), [x, 23, 8], "Second");
						a(fn(x, 23, 8, 22), r, "Third");
						a(i, 2, "Called once");
					}
				};
			}
		};
	},
	Delete: function (a) {
		var i = 0, fn, mfn, x = {};

		fn = function (a, b, c) {
			return a + (++i);
		};
		mfn = memoize(fn);
		a(mfn(3, x, 1), 4, "Init");
		a(mfn(4, x, 1), 6, "Init #2");
		mfn.delete(4, x, 1);
		a(mfn(3, x, 1), 4, "Cached");
		mfn(3, x, 1);
		a(i, 2, "Pre clear");
		mfn.delete(3, x, 1);
		a(i, 2, "After clear");
		a(mfn(3, x, 1), 6, "Reinit");
		a(i, 3, "Reinit count");
		a(mfn(3, x, 1), 6, "Reinit Cached");
		a(i, 3, "Reinit count");
	},
	Clear: function (a) {
		var i = 0, fn, x = {};

		fn = function () {
			++i;
			return arguments;
		};

		fn = memoize(fn, { length: 3 });
		fn(1, x, 3);
		fn(1, x, 4);
		fn(1, x, 3);
		fn(1, x, 4);
		a(i, 2, "Pre clear");
		fn.clear();
		fn(1, x, 3);
		fn(1, x, 4);
		fn(1, x, 3);
		fn(1, x, 4);
		a(i, 4, "After clear");
	}
};
