'use strict';

module.exports = {
	at:                     require('./at'),
	camelToHyphen:          require('./camel-to-hyphen'),
	capitalize:             require('./capitalize'),
	caseInsensitiveCompare: require('./case-insensitive-compare'),
	codePointAt:            require('./code-point-at'),
	contains:               require('./contains'),
	hyphenToCamel:          require('./hyphen-to-camel'),
	endsWith:               require('./ends-with'),
	indent:                 require('./indent'),
	last:                   require('./last'),
	normalize:              require('./normalize'),
	pad:                    require('./pad'),
	plainReplace:           require('./plain-replace'),
	plainReplaceAll:        require('./plain-replace-all'),
	repeat:                 require('./repeat'),
	startsWith:             require('./starts-with')
};
module.exports[require('es6-symbol').iterator] = require('./@@iterator');
