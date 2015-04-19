'use strict';

module.exports = function (t, a) {
	var x = [4, 5, 6], y = { length: 8 }, w = {}, z = { length: 1 };

	a.deep([x, y, w, z].sort(t), [w, z, x, y]);
};
