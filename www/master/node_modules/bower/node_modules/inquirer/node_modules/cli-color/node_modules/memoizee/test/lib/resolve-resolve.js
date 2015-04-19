'use strict';

module.exports = function (t, a) {
	a.deep(t([String, null, Number])([23, 'foo', '45', 'elo']), ['23', 'foo', 45, 'elo']);
};
