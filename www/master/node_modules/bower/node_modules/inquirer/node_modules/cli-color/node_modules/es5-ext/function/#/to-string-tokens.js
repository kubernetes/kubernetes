'use strict';

var validFunction = require('../valid-function')

  , re = new RegExp('^\\s*function[\\0-\'\\)-\\uffff]*' +
	'\\(([\\0-\\(\\*-\\uffff]*)\\)\\s*\\{([\\0-\\uffff]*)\\}\\s*$');

module.exports = function () {
	var data = String(validFunction(this)).match(re);
	return { args: data[1], body: data[2] };
};
