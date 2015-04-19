#!/usr/bin/env node
'use strict';
var pkg = require('./package.json');
var userHome = require('./');

function help() {
	console.log([
		pkg.description,
		'',
		'Example',
		'  $ user-home',
		'  /Users/sindresorhus'
	].join('\n'));
}

if (process.argv.indexOf('--help') !== -1) {
	help();
	return;
}

if (process.argv.indexOf('--version') !== -1) {
	console.log(pkg.version);
	return;
}

process.stdout.write(userHome);
