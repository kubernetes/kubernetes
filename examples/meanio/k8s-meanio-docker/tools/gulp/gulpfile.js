'use strict';

var del = require('del'),
    path = require('path'),
    gulp = require('gulp'),
    gutil = require('gulp-util'),
    through = require('through'),
    gulpLoadPlugins = require('gulp-load-plugins'),
    karma = require('karma').server,
    plugins = gulpLoadPlugins(),
    paths = {
      js: ['*.js', 'test/**/*.js', '!test/coverage/**', '!bower_components/**', 'packages/**/*.js', '!packages/**/node_modules/**', '!packages/contrib/**/*.js', '!packages/contrib/**/node_modules/**'],
      html: ['packages/**/public/**/views/**', 'packages/**/server/views/**'],
      css: ['!bower_components/**', 'packages/**/public/**/css/*.css', '!packages/contrib/**/public/**/css/*.css'],
      less: ['**/public/**/css/*.less']
    },
    assets = require('./config/assets.json'),
    _ = require('lodash');

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

function tokenizeConfig(config) {
  var destTokens = _.keys(config)[0].split('/');

  return {
    srcGlob: _.flatten(_.values(config)),
    destDir: destTokens[destTokens.length - 2],
    destFile: destTokens[destTokens.length - 1]
  };
}

gulp.task('csslint', function () {
  return gulp.src(paths.css)
    .pipe(plugins.csslint('.csslintrc'))
    .pipe(plugins.csslint.reporter())
    .pipe(count('csslint', 'files lint free'));
});

gulp.task('cssmin', function () {
  var config = tokenizeConfig(assets.core.css);

  if (config.srcGlob.length) {
    return gulp.src(config.srcGlob)
      .pipe(plugins.cssmin({keepBreaks: true}))
      .pipe(plugins.concat(config.destFile))
      .pipe(gulp.dest(path.join('bower_components/build', config.destDir)));
  }
});

gulp.task('less', function() {
  return gulp.src(paths.less)
    .pipe(plugins.less())
    .pipe(gulp.dest(function (vinylFile) {
      return vinylFile.cwd;
    }));
});

gulp.task('jshint', function () {
  return gulp.src(paths.js)
    .pipe(plugins.jshint())
    .pipe(plugins.jshint.reporter('jshint-stylish'))
    .pipe(plugins.jshint.reporter('fail'))
    .pipe(count('jshint', 'files lint free'));
});

gulp.task('uglify', function () {
  var config = tokenizeConfig(assets.core.js);

  if (config.srcGlob.length) {
    return gulp.src(config.srcGlob)
      .pipe(plugins.concat(config.destFile))
      .pipe(plugins.uglify({mangle: false}))
      .pipe(gulp.dest(path.join('bower_components/build', config.destDir)));
  }
});

gulp.task('env:test', function () {
  process.env.NODE_ENV = 'test';
});

gulp.task('env:develop', function () {
  process.env.NODE_ENV = 'development';
});

gulp.task('env:production', function () {
  process.env.NODE_ENV = 'production';
});

gulp.task('karma:unit', function (done) {
  karma.start({
    configFile: __dirname + '/karma.conf.js',
    singleRun: true
  }, done);
});

gulp.task('loadTestSchema', function () {
  require('server.js');
  require('meanio/lib/util').preload(__dirname + '/packages/**/server', 'model');
});

gulp.task('mochaTest', ['loadTestSchema'], function () {
  return gulp.src('packages/**/server/tests/**/*.js', {read: false})
    .pipe(plugins.mocha({
      reporter: 'spec'
    }));
});

gulp.task('watch', function () {
  gulp.watch(paths.js, ['jshint']).on('change', plugins.livereload.changed);
  gulp.watch(paths.html).on('change', plugins.livereload.changed);
  gulp.watch(paths.css, ['csslint']).on('change', plugins.livereload.changed);
  gulp.watch(paths.less, ['less']).on('change', plugins.livereload.changed);

  plugins.livereload.listen({interval: 500});
});

// https://github.com/gulpjs/gulp/blob/master/docs/recipes/delete-files-folder.md
gulp.task('clean', function (cb) {
  return del(['bower_components/build'], cb);
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

gulp.task('test', ['env:test', 'karma:unit', 'mochaTest']);

var defaultTasks = ['clean', 'jshint', 'less', 'csslint', 'develop', 'watch'];

if (process.env.NODE_ENV === 'production') {
  defaultTasks = ['clean', 'cssmin', 'uglify'];
}

gulp.task('default', defaultTasks);

// See also: https://github.com/timdp/heroku-buildpack-nodejs-gulp
// For Heroku users only.
// Docs: https://github.com/linnovate/mean/wiki/Deploying-on-Heroku
gulp.task('heroku:production', ['env:production', 'cssmin', 'uglify']);
