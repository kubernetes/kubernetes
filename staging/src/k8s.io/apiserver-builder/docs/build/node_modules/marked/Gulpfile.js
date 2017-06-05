var gulp = require('gulp');
var uglify = require('gulp-uglify');
var concat = require('gulp-concat');

var preserveFirstComment = function() {
  var set = false;

  return function() {
     if (set) return false;
     set = true;
     return true;
  };
};

gulp.task('uglify', function() {
  gulp.src('lib/marked.js')
    .pipe(uglify({preserveComments: preserveFirstComment()}))
    .pipe(concat('marked.min.js'))
    .pipe(gulp.dest('.'));
});

gulp.task('default', ['uglify']);
