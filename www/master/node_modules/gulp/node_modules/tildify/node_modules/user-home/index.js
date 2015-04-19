'use strict';
var env = process.env;
var home = env.HOME;
var user = env.LOGNAME || env.USER || env.LNAME || env.USERNAME;

if (process.platform === 'win32') {
	module.exports = env.USERPROFILE || env.HOMEDRIVE + env.HOMEPATH || home || null;
} else if (process.platform === 'darwin') {
	module.exports = home || (user ? '/Users/' + user : null) || null;
} else if (process.platform === 'linux') {
	module.exports = home ||
		(user ? (process.getuid() === 0 ? '/root' : '/home/' + user) : null) || null;
} else {
	module.exports = home || null;
}
