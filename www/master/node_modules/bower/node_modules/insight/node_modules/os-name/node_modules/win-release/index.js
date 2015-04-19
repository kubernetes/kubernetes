'use strict';
var os = require('os');

var nameMap = {
	'6.4': '10',
	'6.3': '8.1',
	'6.2': '8',
	'6.1': '7',
	'6.0': 'Vista',
	'5.1': 'XP',
	'5.0': '2000',
	'4.9': 'ME',
	'4.1': '98',
	'4.0': '95'
};

module.exports = function (release) {
	return nameMap[(release || os.release()).slice(0, 3)];
};
