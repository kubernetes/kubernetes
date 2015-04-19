'use strict';

module.exports = function (t, a) {
	var fn = function () {};
	a.deep(t(fn), { get: fn, set: fn });
};
