#!/usr/bin/env node
'use strict';
var pkg = require('./package.json');
var osName = require('./');
var argv = process.argv;

function help() {
	console.log([
		'',
		'  ' + pkg.description,
		'',
		'  Example',
		'    os-name',
		'    OS X Mavericks'
	].join('\n'));
}

if (argv.indexOf('--help') !== -1) {
	help();
	return;
}

if (argv.indexOf('--version') !== -1) {
	console.log(pkg.version);
	return;
}

console.log(osName());
