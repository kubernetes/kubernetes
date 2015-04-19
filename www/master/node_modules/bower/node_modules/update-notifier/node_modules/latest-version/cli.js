#!/usr/bin/env node
'use strict';
var pkg = require('./package.json');
var latestVersion = require('./');
var argv = process.argv.slice(2);
var input = argv[0];

function help() {
	console.log([
		'',
		'  ' + pkg.description,
		'',
		'  Usage',
		'    latest-version <package-name>',
		'',
		'  Example',
		'    latest-version pageres',
		'    0.4.1'
	].join('\n'));
}

if (!input || argv.indexOf('--help') !== -1) {
	help();
	return;
}

if (argv.indexOf('--version') !== -1) {
	console.log(pkg.version);
	return;
}

latestVersion(input, function (err, version) {
	if (err) {
		console.error(err);
		process.exit(1);
		return;
	}

	console.log(version);
});
