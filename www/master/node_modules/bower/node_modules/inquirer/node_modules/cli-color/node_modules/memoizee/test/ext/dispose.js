'use strict';

var memoize  = require('../..')
  , nextTick = require('next-tick');

module.exports = function () {
	return {
		Regular: {
			Sync: function (a) {
				var mfn, fn, value = [], x, invoked;
				fn = function (x, y) { return x + y; };
				mfn = memoize(fn, { dispose: function (val) { value.push(val); } });

				mfn(3, 7);
				mfn(5, 8);
				mfn(12, 4);
				a.deep(value, [], "Pre");
				mfn.delete(5, 8);
				a.deep(value, [13], "#1");
				value = [];
				mfn.delete(12, 4);
				a.deep(value, [16], "#2");

				value = [];
				mfn(77, 11);
				mfn.clear();
				a.deep(value, [10, 88], "Clear all");

				x = {};
				invoked = false;
				mfn = memoize(function () { return x; },
					{ dispose: function (val) { invoked = val; } });

				mfn.delete();
				a(invoked, false, "No args: Post invalid delete");
				mfn();
				a(invoked, false, "No args: Post cache");
				mfn.delete();
				a(invoked, x, "No args: Pre delete");
			},
			"Ref counter": function (a) {
				var mfn, fn, value = [];
				fn = function (x, y) { return x + y; };
				mfn = memoize(fn, { refCounter: true,
					dispose: function (val) { value.push(val); } });

				mfn(3, 7);
				mfn(5, 8);
				mfn(12, 4);
				a.deep(value, [], "Pre");
				mfn(5, 8);
				mfn.deleteRef(5, 8);
				a.deep(value, [], "Pre");
				mfn.deleteRef(5, 8);
				a.deep(value, [13], "#1");
				value = [];
				mfn.deleteRef(12, 4);
				a.deep(value, [16], "#2");

				value = [];
				mfn(77, 11);
				mfn.clear();
				a.deep(value, [10, 88], "Clear all");
			},
			Async: function (a, d) {
				var mfn, fn, u = {}, value = [];
				fn = function (x, y, cb) {
					nextTick(function () { cb(null, x + y); });
					return u;
				};

				mfn = memoize(fn, { async: true,
					dispose: function (val) { value.push(val); } });

				mfn(3, 7, function () {
					mfn(5, 8, function () {
						mfn(12, 4, function () {
							a.deep(value, [], "Pre");
							mfn.delete(5, 8);
							a.deep(value, [13], "#1");
							value = [];
							mfn.delete(12, 4);
							a.deep(value, [16], "#2");

							value = [];
							mfn(77, 11, function () {
								mfn.clear();
								a.deep(value, [10, 88], "Clear all");
								d();
							});
						});
					});
				});
			}
		},
		Primitive: {
			Sync: function (a) {
				var mfn, fn, value = [];
				fn = function (x, y) { return x + y; };
				mfn = memoize(fn, { dispose: function (val) { value.push(val); } });

				mfn(3, 7);
				mfn(5, 8);
				mfn(12, 4);
				a.deep(value, [], "Pre");
				mfn.delete(5, 8);
				a.deep(value, [13], "#1");
				value = [];
				mfn.delete(12, 4);
				a.deep(value, [16], "#2");

				value = [];
				mfn(77, 11);
				mfn.clear();
				a.deep(value, [10, 88], "Clear all");
			},
			"Ref counter": function (a) {
				var mfn, fn, value = [];
				fn = function (x, y) { return x + y; };
				mfn = memoize(fn, { refCounter: true,
					dispose: function (val) { value.push(val); } });

				mfn(3, 7);
				mfn(5, 8);
				mfn(12, 4);
				a.deep(value, [], "Pre");
				mfn(5, 8);
				mfn.deleteRef(5, 8);
				a.deep(value, [], "Pre");
				mfn.deleteRef(5, 8);
				a.deep(value, [13], "#1");
				value = [];
				mfn.deleteRef(12, 4);
				a.deep(value, [16], "#2");

				value = [];
				mfn(77, 11);
				mfn.clear();
				a.deep(value, [10, 88], "Clear all");
			},
			Async: function (a, d) {
				var mfn, fn, u = {}, value = [];
				fn = function (x, y, cb) {
					nextTick(function () { cb(null, x + y); });
					return u;
				};

				mfn = memoize(fn, { async: true,
					dispose: function (val) { value.push(val); } });

				mfn(3, 7, function () {
					mfn(5, 8, function () {
						mfn(12, 4, function () {
							a.deep(value, [], "Pre");
							mfn.delete(5, 8);
							a.deep(value, [13], "#1");
							value = [];
							mfn.delete(12, 4);
							a.deep(value, [16], "#2");

							value = [];
							mfn(77, 11, function () {
								mfn.clear();
								a.deep(value, [10, 88], "Clear all");
								d();
							});
						});
					});
				});
			}
		}
	};
};
