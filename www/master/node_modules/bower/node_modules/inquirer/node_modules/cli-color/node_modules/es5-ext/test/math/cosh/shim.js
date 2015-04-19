'use strict';

module.exports = function (t, a) {
	a(t({}), NaN, "NaN");
	a(t(0), 1, "Zero");
	a(t(Infinity), Infinity, "Infinity");
	a(t(-Infinity), Infinity, "-Infinity");
	a(t(1), 1.5430806348152437, "1");
};
