'use strict';

var aFrom   = require('es5-ext/array/from')
  , memoize = require('../..');

module.exports = function () {
	return {
		"": function (a) {
			var i = 0, fn = function () { ++i; return arguments; }, r;

			fn = memoize(fn, { length: false });
			return {
				"No args": function () {
					i = 0;
					a.deep(aFrom(r = fn()), [], "First");
					a(fn(), r, "Second");
					a(fn(), r, "Third");
					a(i, 1, "Called once");
				},
				"Some Args": function () {
					var x = {};
					i = 0;
					a.deep(aFrom(r = fn(x, 8)), [x, 8], "First");
					a(fn(x, 8), r, "Second");
					a(fn(x, 8), r, "Third");
					a(i, 1, "Called once");
				},
				"Many args": function () {
					var x = {};
					i = 0;
					a.deep(aFrom(r = fn(x, 8, 23, 98)), [x, 8, 23, 98], "First");
					a(fn(x, 8, 23, 98), r, "Second");
					a(fn(x, 8, 23, 98), r, "Third");
					a(i, 1, "Called once");
				}
			};
		},
		Delete: function (a) {
			var i = 0, fn, mfn, x = {};

			fn = function (a, b, c) {
				return a + (++i);
			};
			mfn = memoize(fn, { length: false });
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
		}
	};
};
