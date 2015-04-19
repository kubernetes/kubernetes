'use strict';

var getPrototypeOf = Object.getPrototypeOf, prototype = Object.prototype
  , toString = prototype.toString

  , id = Object().toString();

module.exports = function (value) {
	var proto, constructor;
	if (!value || (typeof value !== 'object') || (toString.call(value) !== id)) {
		return false;
	}
	proto = getPrototypeOf(value);
	if (proto === null) {
		constructor = value.constructor;
		if (typeof constructor !== 'function') return true;
		return (constructor.prototype !== value);
	}
	return (proto === prototype) || (getPrototypeOf(proto) === null);
};
