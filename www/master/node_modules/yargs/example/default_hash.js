#!/usr/bin/env node

var argv = require('yargs')
    .default({ x : 10, y : 10 })
    .argv
;

console.log(argv.x + argv.y);
