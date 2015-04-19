var gutil = require('gulp-util');
var exec = require('child_process').exec;
var escape = require('any-shell-escape');

module.exports = function (branch, opt, cb) {
  if(!opt) opt = {};
  if(!branch) return cb(new Error('gulp-git: Branch name is require git.merge("name")'));
  if(!opt.cwd) opt.cwd = process.cwd();
  if(!opt.args) opt.args = ' ';
  
  var cmd = "git merge " + escape([branch]) + " " + opt.args;
  return exec(cmd, {cwd: opt.cwd}, function(err, stdout, stderr){
    if(err) gutil.log(err);
    gutil.log(stdout, stderr);
    if(cb) cb();
  });
};
