var map = require("map-stream");
var gutil = require("gulp-util");
var exec = require("child_process").exec;
var escape = require("any-shell-escape");

module.exports = function (opt) {
  if(!opt) opt = {};
  if(!opt.args) opt.args = ' ';

  function add(file, cb) {
    var cmd = "git add " + escape([file.path]) + " " + opt.args;
    exec(cmd, {cwd: file.cwd}, function(err, stdout, stderr){
      if(err) gutil.log(err);
      gutil.log(stdout, stderr);
      cb(null, file);
    });
  }

  // Return a stream
  return map(add);
};