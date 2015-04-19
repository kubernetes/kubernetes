'use strict';
var userHome = require('user-home');

module.exports = function (str) {
	return str.replace(userHome, '~');
};
