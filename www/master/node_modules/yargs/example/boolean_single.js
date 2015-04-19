#!/usr/bin/env node
var argv = require('yargs')
    .boolean('v')
    .argv
;
console.dir(argv.v);
console.dir(argv._);
