'use strict';
var hasOwnProp = Object.prototype.hasOwnProperty;

module.exports = function (obj, cb) {
	var ret = {};

	for (var key in obj) {
		if (hasOwnProp.call(obj, key)) {
			var res = cb(key, obj[key], obj);
			ret[res[0]] = res[1];
		}
	}

	return ret;
};
