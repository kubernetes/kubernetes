#!/usr/bin/env node
'use strict';
var pkg = require('./package.json');
var opn = require('./');

function help() {
	console.log([
		pkg.description,
		'',
		'Usage',
		'  $ opn <file|url> [app]',
		'',
		'Example',
		'  $ opn http://sindresorhus.com',
		'  $ opn http://sindresorhus.com firefox',
		'  $ opn unicorn.png'
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

opn(process.argv[2], process.argv[3]);
