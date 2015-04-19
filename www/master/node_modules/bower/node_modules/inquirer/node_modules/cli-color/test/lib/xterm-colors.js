'use strict';

module.exports = function (t, a) {
	var re = /^[0-9a-f]{6}$/;

	a(t.length, 256, "Length");
	t.forEach(function (data, index) {
		a(re.test(data), true, "In range #" + index);
	});
};
