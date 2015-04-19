'use strict';

var toString = Object.prototype.toString

  , id = toString.call(new Date());

module.exports = function (x) {
	return (x && ((x instanceof Date) || (toString.call(x) === id))) || false;
};
