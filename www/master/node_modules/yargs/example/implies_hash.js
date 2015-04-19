#!/usr/bin/env node

var argv = require('yargs')
    .usage('Usage: $0 -x [num] -y [num] -w [msg] -h [msg]')
    .implies({
        x: 'y',
        w: '--no-h',
        1: 'h'
    })
    .argv;

if (argv.x) {
    console.log('x / y : ' + (argv.x / argv.y));
}

if (argv.y) {
    console.log('y: ' + argv.y);
}

if (argv.w) {
    console.log('w: ' +argv.w);
}

if (argv.h) {
    console.log('h: ' +argv.h);
}
