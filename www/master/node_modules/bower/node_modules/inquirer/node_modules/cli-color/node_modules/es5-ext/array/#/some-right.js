'use strict';

var callable = require('../../object/valid-callable')
  , value    = require('../../object/valid-value')

  , hasOwnProperty = Object.prototype.hasOwnProperty
  , call = Function.prototype.call;

module.exports = function (cb/*, thisArg*/) {
	var i, self, thisArg;
	self = Object(value(this));
	callable(cb);
	thisArg = arguments[1];

	for (i = self.length; i >= 0; --i) {
		if (hasOwnProperty.call(self, i) &&
				call.call(cb, thisArg, self[i], i, self)) {
			return true;
		}
	}
	return false;
};
