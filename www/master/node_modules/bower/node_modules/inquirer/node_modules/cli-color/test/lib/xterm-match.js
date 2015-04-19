'use strict';

module.exports = function (t, a) {
	a(t.length, 256, "Length");
	t.forEach(function (data, index) {
		a(((data >= 30) && (data <= 37)) || ((data >= 90) && (data <= 97)), true,
			"In range #" + index);
	});
};
