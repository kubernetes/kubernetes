'use strict';
module.exports = function (arr) {
	var rest = [].concat.apply([], [].slice.call(arguments, 1));
	return arr.filter(function (el) {
		return rest.indexOf(el) === -1;
	});
};
