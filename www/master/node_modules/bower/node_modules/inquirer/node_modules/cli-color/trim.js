// Trim formatting from string

'use strict';

var r = new RegExp('\x1b(?:\\[(?:\\d+[ABCDEFGJKSTm]|\\d+;\\d+[Hfm]|' +
	'\\d+;\\d+;\\d+m|6n|s|u|\\?25[lh])|\\w)', 'g');

module.exports = function (str) { return str.replace(r, ''); };
