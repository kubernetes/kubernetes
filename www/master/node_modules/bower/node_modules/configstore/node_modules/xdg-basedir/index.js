'use strict';
var path = require('path');
var userHome = require('user-home');
var env = process.env;

exports.data = env.XDG_DATA_HOME ||
	(userHome ? path.join(userHome, '.local', 'share') : null);

exports.config = env.XDG_CONFIG_HOME ||
	(userHome ? path.join(userHome, '.config') : null);

exports.cache = env.XDG_CONFIG_HOME || (userHome ? path.join(userHome, '.cache') : null);

exports.runtime = env.XDG_RUNTIME_DIR || null;

exports.dataDirs = (env.XDG_DATA_DIRS || '/usr/local/share/:/usr/share/').split(':');

if (exports.data) {
	exports.dataDirs.unshift(exports.data);
}

exports.configDirs = (env.XDG_CONFIG_DIRS || '/etc/xdg').split(':');

if (exports.config) {
	exports.configDirs.unshift(exports.config);
}
