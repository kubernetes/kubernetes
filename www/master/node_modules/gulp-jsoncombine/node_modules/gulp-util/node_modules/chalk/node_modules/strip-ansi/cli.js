#!/usr/bin/env node
'use strict';
var fs = require('fs');
var pkg = require('./package.json');
var strip = require('./');
var input = process.argv[2];

function help() {
	console.log([
		pkg.description,
		'',
		'Usage',
		'  $ strip-ansi <input-file> > <output-file>',
		'  $ cat <input-file> | strip-ansi > <output-file>',
		'',
		'Example',
		'  $ strip-ansi unicorn.txt > unicorn-stripped.txt'
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

if (input) {
	process.stdout.write(strip(fs.readFileSync(input, 'utf8')));
	return;
}

process.stdin.setEncoding('utf8');
process.stdin.on('data', function (data) {
	process.stdout.write(strip(data));
});
