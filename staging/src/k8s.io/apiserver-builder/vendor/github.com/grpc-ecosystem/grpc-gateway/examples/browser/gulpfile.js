"use strict";

var gulp = require('gulp');

var path = require('path');

var bower = require('gulp-bower');
var exit = require('gulp-exit');
var gprocess = require('gulp-process');
var shell = require('gulp-shell');
var jasmineBrowser = require('gulp-jasmine-browser');
var webpack = require('webpack-stream');

gulp.task('bower', function(){
  return bower();
});

gulp.task('server', shell.task([
  'go build -o bin/example-server github.com/grpc-ecosystem/grpc-gateway/examples/server',
]));

gulp.task('gateway', shell.task([
  'go build -o bin/example-gw github.com/grpc-ecosystem/grpc-gateway/examples',
]));

gulp.task('serve-server', ['server'], function(){
  gprocess.start('server-server', 'bin/example-server', [
      '--logtostderr',
  ]);
  gulp.watch('bin/example-server', ['serve-server']);
});

gulp.task('serve-gateway', ['gateway', 'serve-server'], function(){
  gprocess.start('gateway-server', 'bin/example-gw', [
      '--logtostderr', '--swagger_dir', path.join(__dirname, "../examplepb"),
  ]);
  gulp.watch('bin/example-gateway', ['serve-gateway']);
});

gulp.task('backends', ['serve-gateway', 'serve-server']);

var specFiles = ['*.spec.js'];
gulp.task('test', ['backends'], function(done) {
  return gulp.src(specFiles)
    .pipe(webpack({output: {filename: 'spec.js'}}))
    .pipe(jasmineBrowser.specRunner({
      console: true,
      sourceMappedStacktrace: true,
    }))
    .pipe(jasmineBrowser.headless({
      findOpenPort: true,
      catch: true,
      throwFailures: true,
    }))
    .on('error', function(err) {
      done(err);
      process.exit(1);
    })
    .pipe(exit());
});

gulp.task('serve', ['backends'], function(done) {
  var JasminePlugin = require('gulp-jasmine-browser/webpack/jasmine-plugin');
  var plugin = new JasminePlugin();

  return gulp.src(specFiles)
    .pipe(webpack({
      output: {filename: 'spec.js'},
      watch: true,
      plugins: [plugin],
    }))
    .pipe(jasmineBrowser.specRunner({
      sourceMappedStacktrace: true,
    }))
    .pipe(jasmineBrowser.server({
      port: 8000,
      whenReady: plugin.whenReady,
    }));
});

gulp.task('default', ['test']);
