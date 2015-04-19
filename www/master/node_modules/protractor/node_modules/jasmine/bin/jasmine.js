#!/usr/bin/env node

var path = require('path'),
    Command = require('../lib/command.js'),
    command = new Command(path.resolve()),
    Jasmine = require('../lib/jasmine.js'),
    jasmine = new Jasmine({ projectBaseDir: path.resolve() });

command.run(jasmine, process.argv.slice(2));
