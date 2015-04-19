#!/usr/bin/env node
'use strict';
var stdin = require('get-stdin');
var argv = require('minimist')(process.argv.slice(2));
var pkg = require('./package.json');
var indentString = require('./');
var input = argv._;

function help() {
	console.log([
		'',
		'  ' + pkg.description,
		'',
		'  Usage',
		'    indent-string <string> [--indent <string>] [--count <number>]',
		'    cat file.txt | indent-string > indented-file.txt',
		'',
		'  Example',
		'    indent-string "$(printf \'Unicorns\\nRainbows\\n\')" --indent ♥ --count 4',
		'    ♥♥♥♥Unicorns',
		'    ♥♥♥♥Rainbows'
	].join('\n'));
}

function init(data) {
	console.log(indentString(data, argv.indent || ' ', argv.count));
}

if (argv.help) {
	help();
	return;
}

if (argv.version) {
	console.log(pkg.version);
	return;
}

if (process.stdin.isTTY) {
	if (!input) {
		help();
		return;
	}

	init(input[0]);
} else {
	stdin(init);
}
