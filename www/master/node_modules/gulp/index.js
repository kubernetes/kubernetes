'use strict';

var util = require('util');
var Orchestrator = require('orchestrator');
var gutil = require('gulp-util');
var deprecated = require('deprecated');
var vfs = require('vinyl-fs');

function Gulp() {
  Orchestrator.call(this);
}
util.inherits(Gulp, Orchestrator);

Gulp.prototype.task = Gulp.prototype.add;
Gulp.prototype.run = function () {
  // run() is deprecated as of 3.5 and will be removed in 4.0
  // use task dependencies instead

  // impose our opinion of "default" tasks onto orchestrator
  var tasks = arguments.length ? arguments : ['default'];

  this.start.apply(this, tasks);
};

Gulp.prototype.src = vfs.src;
Gulp.prototype.dest = vfs.dest;
Gulp.prototype.watch = function (glob, opt, fn) {
  if (typeof opt === 'function' || Array.isArray(opt)) {
    fn = opt;
    opt = null;
  }

  // array of tasks given
  if (Array.isArray(fn)) {
    return vfs.watch(glob, opt, function () {
      this.start.apply(this, fn);
    }.bind(this));
  }

  return vfs.watch(glob, opt, fn);
};

// let people use this class from our instance
Gulp.prototype.Gulp = Gulp;

// deprecations
deprecated.field('gulp.env has been deprecated. ' +
  'Use your own CLI parser instead. ' +
  'We recommend using yargs or minimist.',
  console.warn,
  Gulp.prototype,
  'env',
  gutil.env
);

Gulp.prototype.run = deprecated.method('gulp.run() has been deprecated. ' +
  'Use task dependencies or gulp.watch task triggering instead.',
  console.warn,
  Gulp.prototype.run
);

var inst = new Gulp();
module.exports = inst;
