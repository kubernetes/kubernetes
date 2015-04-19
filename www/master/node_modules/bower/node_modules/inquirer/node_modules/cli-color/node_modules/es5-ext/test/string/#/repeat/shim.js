'use strict';

module.exports = function (t, a) {
	a(t.call('a', 0), '', "Empty");
	a(t.call('a', 1), 'a', "1");
	a(t.call('\t', 5), '\t\t\t\t\t', "Whitespace");
	a(t.call('raz', 3), 'razrazraz', "Many chars");
};
