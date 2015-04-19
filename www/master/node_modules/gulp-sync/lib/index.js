'use strict';
var groupCount;

groupCount = 0;

module.exports = function(gulp) {
  var async, sync;
  async = function(tasks, prefix) {
    var result, syncCount, task, _i, _len;
    if (prefix == null) {
      prefix = 'sync group';
    }
    if (prefix === 'sync group') {
      prefix += groupCount++;
    }
    result = [];
    syncCount = 0;
    for (_i = 0, _len = tasks.length; _i < _len; _i++) {
      task = tasks[_i];
      if (Array.isArray(task)) {
        task = sync(task, prefix + ':' + syncCount)[0];
        syncCount++;
      }
      result.push(task);
    }
    return result;
  };
  sync = function(tasks, prefix) {
    var deps, index, task, taskName, _fn, _i, _len;
    if (prefix == null) {
      prefix = 'sync group';
    }
    if (prefix === 'sync group') {
      prefix += groupCount++;
    }
    deps = [];
    _fn = function(taskName, deps, task) {
      return gulp.task(taskName, deps, function(cb) {
        var check, onStop;
        check = task.concat();
        gulp.addListener('task_stop', onStop = function(e) {
          var i;
          if (-1 !== (i = check.indexOf(e.task))) {
            check.splice(i, 1);
            if (check.length === 0) {
              gulp.removeListener('task_stop', onStop);
              return cb();
            }
          }
        });
        return gulp.start.apply(gulp, task);
      });
    };
    for (index = _i = 0, _len = tasks.length; _i < _len; index = ++_i) {
      task = tasks[index];
      taskName = prefix + ':' + index;
      if (Array.isArray(task)) {
        task = async(task, taskName);
      } else {
        task = [task];
      }
      _fn(taskName, deps, task);
      deps = [taskName];
    }
    return deps;
  };
  return {
    async: async,
    sync: sync
  };
};
