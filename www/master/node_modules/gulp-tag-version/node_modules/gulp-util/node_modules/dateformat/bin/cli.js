#!/usr/bin/env node
/**
 * dateformat <https://github.com/felixge/node-dateformat>
 *
 * Copyright (c) 2014 Charlike Mike Reagent (cli), contributors.
 * Released under the MIT license.
 */

'use strict';

/**
 * Module dependencies.
 */

var dateFormat = require('../lib/dateformat');
var meow = require('meow');
var stdin = require('get-stdin');

var cli = meow({
  pkg: '../package.json',
  help: [
    'Options',
    '  --help          Show this help',
    '  --version       Current version of package',
    '  -d | --date     Date that want to format (Date object as Number or String)',
    '  -m | --mask     Mask that will use to format the date',
    '  -u | --utc      Convert local time to UTC time or use `UTC:` prefix in mask',
    '  -g | --gmt      You can use `GMT:` prefix in mask',
    '',
    'Usage',
    '  dateformat [date] [mask]',
    '  dateformat "Nov 26 2014" "fullDate"',
    '  dateformat 1416985417095 "dddd, mmmm dS, yyyy, h:MM:ss TT"',
    '  dateformat 1315361943159 "W"',
    '  dateformat "UTC:h:MM:ss TT Z"',
    '  dateformat "longTime" true',
    '  dateformat "longTime" false true',
    '  dateformat "Jun 9 2007" "fullDate" true',
    '  date +%s | dateformat',
    ''
  ].join('\n')
})

var date = cli.input[0] || cli.flags.d || cli.flags.date || Date.now();
var mask = cli.input[1] || cli.flags.m || cli.flags.mask || dateFormat.masks.default;
var utc = cli.input[2] || cli.flags.u || cli.flags.utc || false;
var gmt = cli.input[3] || cli.flags.g || cli.flags.gmt || false;

utc = utc === 'true' ? true : false;
gmt = gmt === 'true' ? true : false;

if (!cli.input.length) {
  stdin(function(date) {
    console.log(dateFormat(date, dateFormat.masks.default, utc, gmt));
  });
  return;
}

if (cli.input.length === 1 && date) {
  mask = date;
  date = Date.now();
  console.log(dateFormat(date, mask, utc, gmt));
  return;
}

if (cli.input.length >= 2 && date && mask) {
  if (mask === 'true' || mask === 'false') {
    utc = mask === 'true' ? true : false;
    gmt = !utc;
    mask = date
    date = Date.now();
  }
  console.log(dateFormat(date, mask, utc, gmt));
  return;
}
