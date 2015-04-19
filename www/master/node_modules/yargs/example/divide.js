#!/usr/bin/env node

var argv = require('yargs')
    .usage('Usage: $0 -x [num] -y [num]')
    .demand(['x','y'])
    .argv;

console.log(argv.x / argv.y);
