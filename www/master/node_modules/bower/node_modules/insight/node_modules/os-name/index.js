'use strict';
var os = require('os');
var osxRelease = require('osx-release');
var winRelease = require('win-release');

module.exports = function (platform, release) {
	if (!platform && release) {
		throw new Error('You can\'t specify a `release` without specfying `platform`');
	}

	platform = platform || os.platform();
	release = release || os.release();

	var id;

	if (platform === 'darwin') {
		id = osxRelease(release).name;
		return 'OS X' + (id ? ' ' + id : '');
	}

	if (platform === 'linux') {
		id = release.replace(/^(\d+\.\d+).*/, '$1');
		return 'Linux' + (id ? ' ' + id : '');
	}

	if (platform === 'win32') {
		id = winRelease(release);
		return 'Windows' + (id ? ' ' + id : '');
	}

	return platform;
};
