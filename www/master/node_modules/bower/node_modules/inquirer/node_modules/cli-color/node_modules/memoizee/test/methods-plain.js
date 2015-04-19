'use strict';

var d = require('d');

require('../ext/dispose');
require('../ext/ref-counter');

module.exports = function (t, a) {
	var value = [], obj = {};
	Object.defineProperties(obj, t({
		someFn: d(function (x, y) { a(this, obj); return x + y; },
			{ refCounter: true,
				dispose: function (val) { value.push(val); } })
	}));

	obj = Object.create(obj);
	obj.someFn(3, 7);
	obj.someFn(5, 8);
	obj.someFn(12, 4);
	a.deep(value, [], "Pre");
	obj.someFn(5, 8);
	obj.someFn.deleteRef(5, 8);
	a.deep(value, [], "Pre");
	obj.someFn.deleteRef(5, 8);
	a.deep(value, [13], "#1");
	value = [];
	obj.someFn.deleteRef(12, 4);
	a.deep(value, [16], "#2");

	value = [];
	obj.someFn(77, 11);
	obj.someFn.clear();
	a.deep(value, [10, 88], "Clear all");
};
