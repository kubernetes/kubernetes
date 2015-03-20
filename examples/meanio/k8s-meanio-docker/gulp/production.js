var gulp = require('gulp');

gulp.task('env:production', function () {
  process.env.NODE_ENV = 'production';
});

if (process.env.NODE_ENV === 'production') {
  defaultTasks = ['clean', 'cssmin', 'uglify'];
}

gulp.task('cssmin', function () {
  var config = tokenizeConfig(assets.core.css);

  if (config.srcGlob.length) {
    return gulp.src(config.srcGlob)
      .pipe(plugins.cssmin({keepBreaks: true}))
      .pipe(plugins.concat(config.destFile))
      .pipe(gulp.dest(path.join('bower_components/build', config.destDir)));
  }
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
