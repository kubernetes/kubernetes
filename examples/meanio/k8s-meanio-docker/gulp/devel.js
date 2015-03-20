'use strict';


var gulp = require('gulp'),
  gulpLoadPlugins = require('gulp-load-plugins');
var del = require('del');
var plugins = gulpLoadPlugins();
var paths = gulp.paths;
//var defaultTasks = ['clean', 'jshint', 'less', 'csslint', 'develop', 'watch'];
gulp.task('help', plugins.taskListing);
var defaultTasks = ['clean', 'jshint', 'csslint','develop','watch'];

gulp.task('clean', function (cb) {

  return del(['bower_components/build'], cb);
});

gulp.task('jshint', function () {
  return gulp.src(paths.js)
    .pipe(plugins.jshint())
    .pipe(plugins.jshint.reporter('jshint-stylish'))
    .pipe(plugins.jshint.reporter('fail'))
    .pipe(count('jshint', 'files lint free'));
});

gulp.task('csslint', function () {
  return gulp.src(paths.css)
    .pipe(plugins.csslint('.csslintrc'))
    .pipe(plugins.csslint.reporter())
    .pipe(count('csslint', 'files lint free'));
});

gulp.task('develop', ['env:develop'], function () {
  plugins.nodemon({
    script: 'server.js',
    ext: 'html js',
    env: { 'NODE_ENV': 'development' } ,
    ignore: ['./node_modules/**'],
    nodeArgs: ['--debug']
  });
});

gulp.task('watch', function () {
  gulp.watch(paths.js, ['jshint']).on('change', plugins.livereload.changed);
  gulp.watch(paths.html).on('change', plugins.livereload.changed);
  gulp.watch(paths.css, ['csslint']).on('change', plugins.livereload.changed);
  gulp.watch(paths.less, ['less']).on('change', plugins.livereload.changed);

  plugins.livereload.listen({interval: 500});
});

gulp.task('default', defaultTasks);

function count(taskName, message) {
  var fileCount = 0;

  function countFiles(file) {
    fileCount++; // jshint ignore:line
  }

  function endStream() {
    gutil.log(gutil.colors.cyan(taskName + ': ') + fileCount + ' ' + message || 'files processed.');
    this.emit('end'); // jshint ignore:line
  }

  return through(countFiles, endStream);
}
