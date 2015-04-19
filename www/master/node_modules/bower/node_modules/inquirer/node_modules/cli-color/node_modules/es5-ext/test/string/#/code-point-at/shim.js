// Taken from: https://github.com/mathiasbynens/String.prototype.codePointAt
//             /blob/master/tests/tests.js

'use strict';

module.exports = function (t, a) {
	a(t.length, 1, "Length");

	// String that starts with a BMP symbol
	a(t.call('abc\uD834\uDF06def', ''), 0x61);
	a(t.call('abc\uD834\uDF06def', '_'), 0x61);
	a(t.call('abc\uD834\uDF06def'), 0x61);
	a(t.call('abc\uD834\uDF06def', -Infinity), undefined);
	a(t.call('abc\uD834\uDF06def', -1), undefined);
	a(t.call('abc\uD834\uDF06def', -0), 0x61);
	a(t.call('abc\uD834\uDF06def', 0), 0x61);
	a(t.call('abc\uD834\uDF06def', 3), 0x1D306);
	a(t.call('abc\uD834\uDF06def', 4), 0xDF06);
	a(t.call('abc\uD834\uDF06def', 5), 0x64);
	a(t.call('abc\uD834\uDF06def', 42), undefined);
	a(t.call('abc\uD834\uDF06def', Infinity), undefined);
	a(t.call('abc\uD834\uDF06def', Infinity), undefined);
	a(t.call('abc\uD834\uDF06def', NaN), 0x61);
	a(t.call('abc\uD834\uDF06def', false), 0x61);
	a(t.call('abc\uD834\uDF06def', null), 0x61);
	a(t.call('abc\uD834\uDF06def', undefined), 0x61);

	// String that starts with an astral symbol
	a(t.call('\uD834\uDF06def', ''), 0x1D306);
	a(t.call('\uD834\uDF06def', '1'), 0xDF06);
	a(t.call('\uD834\uDF06def', '_'), 0x1D306);
	a(t.call('\uD834\uDF06def'), 0x1D306);
	a(t.call('\uD834\uDF06def', -1), undefined);
	a(t.call('\uD834\uDF06def', -0), 0x1D306);
	a(t.call('\uD834\uDF06def', 0), 0x1D306);
	a(t.call('\uD834\uDF06def', 1), 0xDF06);
	a(t.call('\uD834\uDF06def', 42), undefined);
	a(t.call('\uD834\uDF06def', false), 0x1D306);
	a(t.call('\uD834\uDF06def', null), 0x1D306);
	a(t.call('\uD834\uDF06def', undefined), 0x1D306);

	// Lone high surrogates
	a(t.call('\uD834abc', ''), 0xD834);
	a(t.call('\uD834abc', '_'), 0xD834);
	a(t.call('\uD834abc'), 0xD834);
	a(t.call('\uD834abc', -1), undefined);
	a(t.call('\uD834abc', -0), 0xD834);
	a(t.call('\uD834abc', 0), 0xD834);
	a(t.call('\uD834abc', false), 0xD834);
	a(t.call('\uD834abc', NaN), 0xD834);
	a(t.call('\uD834abc', null), 0xD834);
	a(t.call('\uD834abc', undefined), 0xD834);

	// Lone low surrogates
	a(t.call('\uDF06abc', ''), 0xDF06);
	a(t.call('\uDF06abc', '_'), 0xDF06);
	a(t.call('\uDF06abc'), 0xDF06);
	a(t.call('\uDF06abc', -1), undefined);
	a(t.call('\uDF06abc', -0), 0xDF06);
	a(t.call('\uDF06abc', 0), 0xDF06);
	a(t.call('\uDF06abc', false), 0xDF06);
	a(t.call('\uDF06abc', NaN), 0xDF06);
	a(t.call('\uDF06abc', null), 0xDF06);
	a(t.call('\uDF06abc', undefined), 0xDF06);

	a.throws(function () { t.call(undefined); }, TypeError);
	a.throws(function () { t.call(undefined, 4); }, TypeError);
	a.throws(function () { t.call(null); }, TypeError);
	a.throws(function () { t.call(null, 4); }, TypeError);
	a(t.call(42, 0), 0x34);
	a(t.call(42, 1), 0x32);
	a(t.call({ toString: function () { return 'abc'; } }, 2), 0x63);

	a.throws(function () { t.apply(undefined); }, TypeError);
	a.throws(function () { t.apply(undefined, [4]); }, TypeError);
	a.throws(function () { t.apply(null); }, TypeError);
	a.throws(function () { t.apply(null, [4]); }, TypeError);
	a(t.apply(42, [0]), 0x34);
	a(t.apply(42, [1]), 0x32);
	a(t.apply({ toString: function () { return 'abc'; } }, [2]), 0x63);
};
