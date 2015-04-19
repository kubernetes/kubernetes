'use strict';

module.exports = function (t, a) {
	a(t({}), NaN, "NaN");
	a(t(-2), NaN, "Less than -1");
	a(t(2), NaN, "Greater than 1");
	a(t(-1), -Infinity, "-1");
	a(t(1), Infinity, "1");
	a(t(0), 0, "Zero");
	a(t(0.5), 0.5493061443340549, "Ohter");
};
