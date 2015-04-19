'use strict';

var o = [1, 2, [3, 4, [5, 6], 7, 8], 9, 10];

module.exports = {
	__generic: function (t, a) {
		a(t.call(this).length, 3);
	},
	"Nested Arrays": function (t, a) {
		a(t.call(o).length, 10);
	}
};
