'use strict';
module.exports = function (url) {
	if (typeof url !== 'string') {
		throw new TypeError('Expected a string');
	}

	return url.trim().replace(/^(?!(?:\w+:)?\/\/)/, 'http://');
};
