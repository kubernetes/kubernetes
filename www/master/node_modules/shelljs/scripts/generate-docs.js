#!/usr/bin/env node
require('../global');

echo('Appending docs to README.md');

cd(__dirname + '/..');

// Extract docs from shell.js
var docs = grep('//@', 'shell.js');

docs = docs.replace(/\/\/\@include (.+)/g, function(match, path) {
  var file = path.match('.js$') ? path : path+'.js';
  return grep('//@', file);
});

// Remove '//@'
docs = docs.replace(/\/\/\@ ?/g, '');
// Append docs to README
sed('-i', /## Command reference(.|\n)*/, '## Command reference\n\n' + docs, 'README.md');

echo('All done.');
