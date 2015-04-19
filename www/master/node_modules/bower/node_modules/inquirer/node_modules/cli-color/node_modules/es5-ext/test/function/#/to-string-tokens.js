'use strict';

module.exports = function (t, a) {
	a.deep(t.call(function (a, b) { return this[a] + this[b]; }),
		{ args: 'a, b', body: ' return this[a] + this[b]; ' });
	a.deep(t.call(function () {}),
		{ args: '', body: '' });
	a.deep(t.call(function (raz) {}),
		{ args: 'raz', body: '' });
	a.deep(t.call(function () { Object(); }),
		{ args: '', body: ' Object(); ' });
};
