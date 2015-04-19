'use strict';
module.exports = function (scope) {
	var rc = require('rc')('npm', {registry: 'https://registry.npmjs.org/'});
	return rc[scope + ':registry'] || (rc.registry.slice(-1) !== '/' ? rc.registry + '/' : rc.registry);
};
