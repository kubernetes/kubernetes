var gulp = require('gulp');
var git = require('../');



// Init a git repo

gulp.task('init', function(){
  git.init();
});


// Add files

gulp.task('add', function(){
  gulp.src('./*')
  .pipe(git.add());
});


// Commit files

gulp.task('commit', function(){
  gulp.src('./*', {buffer:false})
  .pipe(git.commit('initial commit', {args: "-v"}));
});

// Commit files with arguments
gulp.task('commitopts', function(){
  gulp.src('./*')
  .pipe(git.commit('initial commit', {args: "-v"}));
});

// Commit files with templates
gulp.task('committemplate', function(){
  gulp.src('./*')
  .pipe(git.commit('initial commit file: <%= file.path%>', {args: "-v"}));
});


// Add remote

gulp.task('remote', function(){
  git.addRemote('origin', 'https://github.com/stevelacy/git-test');
});


// Push to remote repo

gulp.task('push', function(){
  git.push('origin', 'master');
});


// Pull from remote repo

gulp.task('pull', function(){
  git.pull('origin', 'master');
});

// Tag the repo

gulp.task('tag', function(){
  git.tag('v1.1.1', 'Version message');
});

// Tag the repo WITH signed key
gulp.task('tagsec', function(){
  git.tag('v1.1.1', 'Version message with signed key', true);
});

gulp.task('rm', function(){
  gulp.src('./delete')
  .pipe(git.rm());
});


// default gulp task

gulp.task('default', ['add']);
