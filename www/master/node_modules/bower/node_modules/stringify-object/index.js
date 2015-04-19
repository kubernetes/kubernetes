'use strict';

function isObject(val) {
	return val === Object(val);
}

function isEmpty(val) {
	if (val === undefined || val === null) {
		return true;
	}

	if (Array.isArray(val) || typeof val === 'string') {
		return val.length === 0;
	}

	for (var key in val) {
		if (Object.prototype.hasOwnProperty.call(val, key)) {
			return false;
		}
	}

	return true;
}

module.exports = function (val, opts, pad) {
	var cache = [];

	return (function stringify(val, opts, pad) {
		var objKeys;
		opts = opts || {};
		opts.indent = opts.indent || '\t';
		pad = pad || '';

		if (typeof val === 'number' ||
			typeof val === 'boolean' ||
			val === null ||
			val === undefined) {
			return val;
		}

		if (val instanceof Date) {
			return "new Date('" + val.toISOString() + "')";
		}

		if (Array.isArray(val)) {
			if (isEmpty(val)) {
				return '[]';
			}

			return '[\n' + val.map(function (el, i) {
				var eol = val.length - 1 === i ? '\n' : ',\n';
				return pad + opts.indent + stringify(el, opts, pad + opts.indent) + eol;
			}).join('') + pad + ']';
		}

		if (isObject(val)) {
			if (cache.indexOf(val) !== -1) {
				return null;
			}

			if (isEmpty(val)) {
				return '{}';
			}

			cache.push(val);
			objKeys = Object.keys(val);

			return '{\n' + objKeys.map(function (el, i) {
				var eol = objKeys.length - 1 === i ? '\n' : ',\n';
				var key = /^[a-z$_][a-z$_0-9]*$/i.test(el) ? el : stringify(el, opts);
				return pad + opts.indent + key + ': ' + stringify(val[el], opts, pad + opts.indent) + eol;
			}).join('') + pad + '}';
		}

		if (opts.singleQuotes === false) {
			return '"' + val.replace(/"/g, '\\\"') + '"';
		} else {
			return "'" + val.replace(/'/g, "\\\'") + "'";
		}
	})(val, opts, pad);
};
