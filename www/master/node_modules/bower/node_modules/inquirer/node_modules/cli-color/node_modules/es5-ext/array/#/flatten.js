'use strict';

var isArray = Array.isArray, forEach = Array.prototype.forEach
  , push = Array.prototype.push;

module.exports = function flatten() {
	var r = [];
	forEach.call(this, function (x) {
		push.apply(r, isArray(x) ? flatten.call(x) : [x]);
	});
	return r;
};
