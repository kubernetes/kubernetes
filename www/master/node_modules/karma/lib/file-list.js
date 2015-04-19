var fs = require('fs');
var glob = require('glob');
var mm = require('minimatch');
var q = require('q');

var helper = require('./helper');
var log = require('./logger').create('watcher');


var createWinGlob = function(realGlob) {
  return function(pattern, options, done) {
    realGlob(pattern, options, function(err, results) {
      done(err, results.map(helper.normalizeWinPath));
    });
  };
};

if (process.platform === 'win32') {
  glob = createWinGlob(glob);
}


var File = function(path, mtime) {
  // used for serving (processed path, eg some/file.coffee -> some/file.coffee.js)
  this.path = path;

  // original absolute path, id of the file
  this.originalPath = path;

  // where the content is stored (processed)
  this.contentPath = path;

  this.mtime = mtime;
  this.isUrl = false;
};

var Url = function(path) {
  this.path = path;
  this.isUrl = true;
};

Url.prototype.toString = File.prototype.toString = function() {
  return this.path;
};


var GLOB_OPTS = {
  // globDebug: true,
  cwd: '/'
};


var byPath = function(a, b) {
  if (a.path > b.path) {
    return 1;
  }
  if (a.path < b.path) {
    return -1;
  }
  return 0;
};


// TODO(vojta): ignore changes (add/change/remove) when in the middle of refresh
// TODO(vojta): do not glob patterns that are watched (both on init and refresh)
var List = function(patterns, excludes, emitter, preprocess, batchInterval) {
  var self = this;
  var pendingDeferred;
  var pendingTimeout;

  var resolveFiles = function(buckets) {
    var uniqueMap = {};
    var files = {
      served: [],
      included: []
    };

    buckets.forEach(function(bucket, idx) {
      bucket.sort(byPath).forEach(function(file) {
        if (!uniqueMap[file.path]) {
          if (patterns[idx].served) {
            files.served.push(file);
          }

          if (patterns[idx].included) {
            files.included.push(file);
          }

          uniqueMap[file.path] = true;
        }
      });
    });

    return files;
  };

  var resolveDeferred = function(files) {
    if (pendingTimeout) {
      clearTimeout(pendingTimeout);
    }

    pendingDeferred.resolve(files || resolveFiles(self.buckets));
    pendingDeferred = pendingTimeout = null;
  };

  var fireEventAndDefer = function() {
    if (pendingTimeout) {
      clearTimeout(pendingTimeout);
    }

    if (!pendingDeferred) {
      pendingDeferred = q.defer();
      emitter.emit('file_list_modified', pendingDeferred.promise);
    }

    pendingTimeout = setTimeout(resolveDeferred, batchInterval);
  };


  // re-glob all the patterns
  this.refresh = function() {
    // TODO(vojta): cancel refresh if another refresh starts
    var buckets = self.buckets = new Array(patterns.length);

    var complete = function() {
      if (buckets !== self.buckets) {
        return;
      }

      var files = resolveFiles(buckets);

      resolveDeferred(files);
      log.debug('Resolved files:\n\t' + files.served.join('\n\t'));
    };

    // TODO(vojta): use some async helper library for this
    var pending = 0;
    var finish = function() {
      pending--;

      if (!pending) {
        complete();
      }
    };

    if (!pendingDeferred) {
      pendingDeferred = q.defer();
      emitter.emit('file_list_modified', pendingDeferred.promise);
    }

    if (pendingTimeout) {
      clearTimeout(pendingTimeout);
    }

    patterns.forEach(function(patternObject, i) {
      var pattern = patternObject.pattern;

      if (helper.isUrlAbsolute(pattern)) {
        buckets[i] = [new Url(pattern)];
        return;
      }

      pending++;
      glob(pattern, GLOB_OPTS, function(err, resolvedFiles) {
        var matchedAndNotIgnored = 0;

        buckets[i] = [];

        if (!resolvedFiles.length) {
          log.warn('Pattern "%s" does not match any file.', pattern);
          return finish();
        }

        // stat each file to get mtime and isDirectory
        resolvedFiles.forEach(function(path) {
          var matchExclude = function(excludePattern) {
            return mm(path, excludePattern);
          };

          if (excludes.some(matchExclude)) {
            log.debug('Excluded file "%s"', path);
            return;
          }

          pending++;
          matchedAndNotIgnored++;
          fs.stat(path, function(error, stat) {
            if (error) {
              log.debug('An error occured while reading "%s"', path);
              finish();
            } else {
              if (!stat.isDirectory()) {
                // TODO(vojta): reuse file objects
                var file = new File(path, stat.mtime);

                preprocess(file, function() {
                  buckets[i].push(file);
                  finish();
                });
              } else {
                log.debug('Ignored directory "%s"', path);
                finish();
              }
            }
          });
        });

        if (!matchedAndNotIgnored) {
          log.warn('All files matched by "%s" were excluded.', pattern);
        }

        finish();
      });
    });

    if (!pending) {
      process.nextTick(complete);
    }

    return pendingDeferred.promise;
  };


  // set new patterns and excludes
  // and re-glob
  this.reload = function(newPatterns, newExcludes) {
    patterns = newPatterns;
    excludes = newExcludes;

    return this.refresh();
  };


  /**
   * Adds a new file into the list (called by watcher)
   *  - ignore excluded files
   *  - ignore files that are already in the list
   *  - get mtime (by stat)
   *  - fires "file_list_modified"
   */
  this.addFile = function(path, done) {
    var buckets = this.buckets;
    var i, j;

    // sorry, this callback is just for easier testing
    done = done || function() {};

    // check excludes
    for (i = 0; i < excludes.length; i++) {
      if (mm(path, excludes[i])) {
        log.debug('Add file "%s" ignored. Excluded by "%s".', path, excludes[i]);
        return done();
      }
    }

    for (i = 0; i < patterns.length; i++) {
      if (mm(path, patterns[i].pattern)) {
        for (j = 0; j < buckets[i].length; j++) {
          if (buckets[i][j].originalPath === path) {
            log.debug('Add file "%s" ignored. Already in the list.', path);
            return done();
          }
        }

        break;
      }
    }

    if (i >= patterns.length) {
      log.debug('Add file "%s" ignored. Does not match any pattern.', path);
      return done();
    }

    var addedFile = new File(path);
    buckets[i].push(addedFile);

    return fs.stat(path, function(err, stat) {
      // in the case someone refresh() the list before stat callback
      if (self.buckets === buckets) {
        addedFile.mtime = stat.mtime;

        return preprocess(addedFile, function() {
          // TODO(vojta): ignore if refresh/reload happens
          log.info('Added file "%s".', path);
          fireEventAndDefer();
          done();
        });
      }

      return done();
    });
  };


  /**
   * Update mtime of a file (called by watcher)
   *  - ignore if file is not in the list
   *  - ignore if mtime has not changed
   *  - fire "file_list_modified"
   */
  this.changeFile = function(path, done) {
    var buckets = this.buckets;
    var i, j;

    // sorry, this callback is just for easier testing
    done = done || function() {};

    outer:
    for (i = 0; i < buckets.length; i++) {
      for (j = 0; j < buckets[i].length; j++) {
        if (buckets[i][j].originalPath === path) {
          break outer;
        }
      }
    }

    if (!buckets[i]) {
      log.debug('Changed file "%s" ignored. Does not match any file in the list.', path);
      return done();
    }

    var changedFile = buckets[i][j];
    return fs.stat(path, function(err, stat) {
      // https://github.com/paulmillr/chokidar/issues/11
      if (err || !stat) {
        return self.removeFile(path, done);
      }

      if (self.buckets === buckets && stat.mtime > changedFile.mtime) {
        log.info('Changed file "%s".', path);
        changedFile.mtime = stat.mtime;
        // TODO(vojta): THIS CAN MAKE FILES INCONSISTENT
        // if batched change is resolved before preprocessing is finished, the file can be in
        // inconsistent state, when the promise is resolved.
        // Solutions:
        // 1/ the preprocessor should not change the object in place, but create a copy that would
        // be eventually merged into the original file, here in the callback, synchronously.
        // 2/ delay the promise resolution - wait for any changeFile operations to finish
        return preprocess(changedFile, function() {
          // TODO(vojta): ignore if refresh/reload happens
          fireEventAndDefer();
          done();
        });
      }

      return done();
    });
  };


  /**
   * Remove a file from the list (called by watcher)
   *  - ignore if file is not in the list
   *  - fire "file_list_modified"
   */
  this.removeFile = function(path, done) {
    var buckets = this.buckets;

    // sorry, this callback is just for easier testing
    done = done || function() {};

    for (var i = 0; i < buckets.length; i++) {
      for (var j = 0; j < buckets[i].length; j++) {
        if (buckets[i][j].originalPath === path) {
          buckets[i].splice(j, 1);
          log.info('Removed file "%s".', path);
          fireEventAndDefer();
          return done();
        }
      }
    }

    log.debug('Removed file "%s" ignored. Does not match any file in the list.', path);
    return done();
  };
};
List.$inject = ['config.files', 'config.exclude', 'emitter', 'preprocess',
    'config.autoWatchBatchDelay'];

// PUBLIC
exports.List = List;
exports.File = File;
exports.Url = Url;
