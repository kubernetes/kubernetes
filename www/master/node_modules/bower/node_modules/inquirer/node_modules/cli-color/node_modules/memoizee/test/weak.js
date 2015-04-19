'use strict';

module.exports = function (t, a) {
	var value = [], obj = {}, memoized, count = 0, x, y, z;
	memoized = t(function (arg, x, y) { a(arg, obj); return x + y; },
		{ refCounter: true, dispose: function (val) { value.push(val); } });

	a(memoized(obj, 3, 7), 10);
	a(memoized(obj, 5, 8), 13);
	a(memoized(obj, 12, 4), 16);
	a.deep(value, [], "Pre");
	a(memoized(obj, 5, 8), 13);
	memoized.deleteRef(obj, 5, 8);
	a.deep(value, [], "Pre");
	memoized.deleteRef(obj, 5, 8);
	a.deep(value, [13], "#1");
	value = [];
	memoized.deleteRef(obj, 12, 4);
	a.deep(value, [16], "#2");

	value = [];
	memoized(obj, 77, 11);

	x = {};
	y = {};
	z = {};
	memoized = t(function (arg) { return ++count; });
	a(memoized(x), 1);
	a(memoized(y), 2);
	a(memoized(x), 1);
	a(memoized(z), 3);
	a(count, 3);
};
