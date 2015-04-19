// Thanks for hints: https://github.com/paulmillr/es6-shim

'use strict';

var some = Array.prototype.some, abs = Math.abs, sqrt = Math.sqrt

  , compare = function (a, b) { return b - a; }
  , divide = function (x) { return x / this; }
  , add = function (sum, number) { return sum + number * number; };

module.exports = function (val1, val2/*, …valn*/) {
	var result, numbers;
	if (!arguments.length) return 0;
	some.call(arguments, function (val) {
		if (isNaN(val)) {
			result = NaN;
			return;
		}
		if (!isFinite(val)) {
			result = Infinity;
			return true;
		}
		if (result !== undefined) return;
		val = Number(val);
		if (val === 0) return;
		if (!numbers) numbers = [abs(val)];
		else numbers.push(abs(val));
	});
	if (result !== undefined) return result;
	if (!numbers) return 0;

	numbers.sort(compare);
	return numbers[0] * sqrt(numbers.map(divide, numbers[0]).reduce(add, 0));
};
