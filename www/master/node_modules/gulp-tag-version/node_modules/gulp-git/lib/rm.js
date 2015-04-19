var map = require('map-stream');
var gutil = require('gulp-util');
var exec = require('child_process').exec;
var escape = require('any-shell-escape');

module.exports = function (opt) {
  if(!opt) opt = {};
  if(!opt.args) opt.args = ' ';
  
  function rm(file, cb) {
    console.log(file.path);
    if(!file.path) throw new Error('gulp-git: file is required');
    var cmd = "git rm " + escape([file.path]) + " " + opt.args;
    exec(cmd, {cwd: file.cwd}, function(err, stdout, stderr){
      if(err) gutil.log(err);
      gutil.log(stdout, stderr);
      cb(null, file);
    });
  }

  // Return a stream
  return map(rm);
};