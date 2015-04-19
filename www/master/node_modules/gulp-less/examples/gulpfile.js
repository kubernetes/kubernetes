var gulp = require('gulp');
var less = require('../');

gulp.task('less', function(){
  gulp.src('./normal.less')
  .pipe(less())
  .pipe(gulp.dest('build'));
});

gulp.task('default', ['less']);
