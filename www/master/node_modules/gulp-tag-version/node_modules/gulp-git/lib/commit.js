var through = require('through2');
var gutil = require('gulp-util');
var exec = require('child_process').exec;
var escape = require('any-shell-escape');

module.exports = function (message, opt) {
  if(!opt) opt = {};
  if(!message) throw new Error('gulp-git: Commit message is required git.commit("commit message")');
  if(!opt.args) opt.args = ' ';

  var files = [];
  var cwd = '';

  var stream = through.obj(function(file, enc, cb){
    this.push(file);
    files.push(file.path);
    cwd = file.cwd;
    cb();
  }).on('data', function(){

  }).on('end', function(){
    var cmd = 'git commit -m "' + message + '" ' + escape(files) + ' ' + opt.args;
      exec(cmd, {cwd: cwd}, function(err, stdout, stderr){
        if(err) gutil.log(err);
        gutil.log(stdout, stderr);
    });
  });

  return stream;
};