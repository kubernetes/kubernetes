'use strict';

module.exports = function (t, a) {
	var result = ['foo'];
	result.index = 0;
	result.input = 'foobar';
	a.deep(t.call(/foo/, 'foobar'), result);
};
