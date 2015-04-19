'use strict';
var minimatch = require('minimatch');
var union = require('array-union');
var diff = require('array-differ');

function arrayify(arr) {
	return Array.isArray(arr) ? arr : [arr];
}

module.exports = function (list, patterns, options) {
	list = arrayify(list);
	patterns = arrayify(patterns);

	if (list.length === 0 || patterns.length === 0) {
		return [];
	}

	options = options || {};

	return patterns.reduce(function (ret, pattern, i) {
		if (pattern[0] === '!') {
			return diff(ret, minimatch.match(ret, pattern.slice(1), options));
		}

		return union(ret, minimatch.match(list, pattern, options));
	}, []);
};
