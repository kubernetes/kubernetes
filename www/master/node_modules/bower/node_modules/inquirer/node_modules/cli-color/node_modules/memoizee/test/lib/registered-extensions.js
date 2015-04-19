'use strict';

module.exports = function (t, a) {
	require('../../ext/async');
	a(typeof t.async, 'function');
};
