'use strict';

var isCallable = require('../object/is-callable')
  , value      = require('../object/valid-value')

  , call = Function.prototype.call;

module.exports = function (fmap) {
	fmap = Object(value(fmap));
	return function (pattern) {
		var context = value(this);
		pattern = String(pattern);
		return pattern.replace(/%([a-zA-Z]+)|\\([\u0000-\uffff])/g,
			function (match, token, escape) {
				var t, r;
				if (escape) return escape;
				t = token;
				while (t && !(r = fmap[t])) t = t.slice(0, -1);
				if (!r) return match;
				if (isCallable(r)) r = call.call(r, context);
				return r + token.slice(t.length);
			});
	};
};
