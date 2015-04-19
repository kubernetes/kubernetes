'use strict';

var indexOf = require('./e-index-of')

  , forEach = Array.prototype.forEach, splice = Array.prototype.splice;

module.exports = function (item/*, …item*/) {
	forEach.call(arguments, function (item) {
		var index = indexOf.call(this, item);
		if (index !== -1) splice.call(this, index, 1);
	}, this);
};
