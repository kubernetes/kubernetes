'use strict';

var callable = require('../../../object/valid-callable')
  , value    = require('../../../object/valid-value')

  , some = Array.prototype.some, apply = Function.prototype.apply;

module.exports = function (predicate/*, thisArg*/) {
	var k, self;
	self = Object(value(this));
	callable(predicate);

	return some.call(self, function (value, index) {
		if (apply.call(predicate, this, arguments)) {
			k = index;
			return true;
		}
		return false;
	}, arguments[1]) ? k : -1;
};
