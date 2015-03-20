'use strict';

var del = require('del'),
  path = require('path'),
  gulp = require('gulp'),
  util = require('util'),
  through = require('through'),
  karma = require('karma').server,
  paths = {
    js: ['*.js', 'test/**/*.js', '!test/coverage/**', '!bower_components/**', 'packages/**/*.js', '!packages/**/node_modules/**', '!packages/contrib/**/*.js', '!packages/contrib/**/node_modules/**'],
    html: ['packages/**/public/**/views/**', 'packages/**/server/views/**'],
    css: ['!bower_components/**', 'packages/**/public/**/css/*.css', '!packages/contrib/**/public/**/css/*.css'],
    less: ['**/public/**/css/*.less']
  };
//var assets = require('./config/assets.json');

  var paths = gulp.paths;

gulp.task('env:develop', function () {
  process.env.NODE_ENV = 'development';
});






