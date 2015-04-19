'use strict';

module.exports = function (t, a) {
	var x = [1, 2, {}, 4];
	a(t.call(x), x, "Returns same array");
	a.deep(x, [], "Empties array");
};
