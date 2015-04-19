#!/usr/bin/env node
'use strict';
var pkg = require('./package.json');
var hasAnsi = require('./');
var input = process.argv[2];

function stdin(cb) {
	var ret = '';
	process.stdin.setEncoding('utf8');
	process.stdin.on('data', function (data) {
		ret += data;
	});
	process.stdin.on('end', function () {
		cb(ret);
	});
}

function help() {
	console.log([
		pkg.description,
		'',
		'Usage',
		'  $ has-ansi <string>',
		'  $ echo <string> | has-ansi',
		'',
		'Exits with code 0 if input has ANSI escape codes and 1 if not'
	].join('\n'));
}

function init(data) {
	process.exit(hasAnsi(data) ? 0 : 1);
}

if (process.argv.indexOf('--help') !== -1) {
	help();
	return;
}

if (process.argv.indexOf('--version') !== -1) {
	console.log(pkg.version);
	return;
}

if (process.stdin.isTTY) {
	if (!input) {
		help();
		return;
	}

	init(input);
} else {
	stdin(init);
}
