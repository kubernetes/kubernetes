#!/usr/bin/env node
'use strict';
var pkg = require('./package.json');
var supportsColor = require('./');
var input = process.argv[2];

function help() {
	console.log([
		pkg.description,
		'',
		'Usage',
		'  $ supports-color',
		'',
		'Exits with code 0 if color is supported and 1 if not'
	].join('\n'));
}

if (!input || process.argv.indexOf('--help') !== -1) {
	help();
	return;
}

if (process.argv.indexOf('--version') !== -1) {
	console.log(pkg.version);
	return;
}

process.exit(supportsColor ? 0 : 1);
