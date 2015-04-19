#!/usr/bin/env node
'use strict';
var fs = require('fs');
var pkg = require('./package.json');
var stripBom = require('./');
var argv = process.argv.slice(2);
var input = argv[0];

function help() {
	console.log([
		'',
		'  ' + pkg.description,
		'',
		'  Usage',
		'    strip-bom <file> > <new-file>',
		'    cat <file> | strip-bom > <new-file>',
		'',
		'  Example',
		'    strip-bom unicorn.txt > unicorn-without-bom.txt'
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

if (process.stdin.isTTY) {
	if (!input) {
		help();
		return;
	}

	fs.createReadStream(input).pipe(stripBom.stream()).pipe(process.stdout);
} else {
	process.stdin.pipe(stripBom.stream()).pipe(process.stdout);
}
