#!/usr/bin/env node
'use strict';
var meow = require('meow');
var repeating = require('./');

var cli = meow({
	help: [
		'Usage',
		'  repeating <string> <count>',
		'',
		'Example',
		'  repeating unicorn 2',
		'  unicornunicorn'
	].join('\n')
});

if (typeof cli.input[1] !== 'number') {
	console.error('You have to define how many times to repeat the string.');
	process.exit(1);
}

console.log(repeating(String(cli.input[0]), cli.input[1]));
