'use strict';

var toString = Object.prototype.toString

  , id = toString.call(/a/);

module.exports = function (x) {
	return (x && (x instanceof RegExp || (toString.call(x) === id))) || false;
};
