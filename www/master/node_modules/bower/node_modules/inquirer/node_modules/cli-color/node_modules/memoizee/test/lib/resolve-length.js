'use strict';

module.exports = function (t, a) {
	a(t(1, 2), 1, "Options");
	a(t(1, 2, true), 1, "Options: Async ");
	a(t(undefined, 2), 2, "Function");
	a(t(undefined, 2, true), 1, "Function: Async");
	a(t(undefined, undefined, false), 1, "Unknown");
	a(t(undefined, undefined, true), 1, "Unknown: async");
};
