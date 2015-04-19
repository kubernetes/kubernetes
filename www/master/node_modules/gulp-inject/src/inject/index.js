'use strict';

var fs = require('fs');
var es = require('event-stream');
var path = require('path');
var gutil = require('gulp-util');
var extname = require('../extname');
var transform = require('../transform');
var tags = require('../tags');
var PluginError = gutil.PluginError;
var File = gutil.File;
var magenta = gutil.colors.magenta;
var cyan = gutil.colors.cyan;
var red = gutil.colors.red;

/**
 * Constants
 */
var PLUGIN_NAME = 'gulp-inject';

module.exports = exports = function(sources, opt){
  if (!sources) {
    throw error('Missing sources stream!');
  }
  if (!opt) {
    opt = {};
  }

  if (opt.sort) {
    throw error('sort option is deprecated! Use `sort-stream` module instead!');
  }
  if (opt.templateString) {
    throw error('`templateString` option is deprecated! Create a virtual `vinyl` file instead!');
  }
  if (opt.transform && typeof opt.transform !== 'function') {
    throw error('transform option must be a function');
  }

  // Notify people of common mistakes...
  if (opt.read) {
    warn('There is no ' + magenta('`read`') + ' option. Did you mean to provide it for ' + magenta('`gulp.src`') + ' perhaps?');
  }

  // Defaults:
  opt.ignorePath = toArray(opt.ignorePath).map(unixify);
  opt.relative = bool(opt, 'relative', false);
  opt.addRootSlash = bool(opt, 'addRootSlash', !opt.relative);
  opt.transform = defaults(opt, 'transform', transform);
  opt.tags = tags();
  opt.tags.name = defaults(opt, 'name', 'inject');
  transform.selfClosingTag = bool(opt, 'selfClosingTag', false);

  // Is the first parameter a Vinyl File Stream:
  if (typeof sources.on === 'function' && typeof sources.pipe === 'function') {
    return handleVinylStream(sources, opt);
  }

  throw error('passing target file as a string is deprecated! Pass a vinyl file stream (i.e. use `gulp.src`)!');
};

function defaults (options, prop, defaultValue) {
  return options[prop] || defaultValue;
}

function bool (options, prop, defaultVal) {
  return typeof options[prop] !== 'undefined' ? !!options[prop] : defaultVal;
}

/**
 * Handle injection when files to
 * inject comes from a Vinyl File Stream
 *
 * @param {Stream} sources
 * @param {Object} opt
 * @returns {Stream}
 */
function handleVinylStream (sources, opt) {
  var collected = collectFilesToInject(sources, opt);

  return es.map(function (target, cb) {
    if (target.isStream()) {
      return cb(error('Streams not supported for target templates!'));
    }
    collected(function (collection) {
      target.contents = getNewContent(target, collection, opt);
      cb(null, target);
    });
  });
}

/**
 * Collecting files to inject from Vinyl File Stream
 *
 * Returns an almost promise like function which can be
 * called multiple times with a callback, that will be
 * resolved with the result of the file collection.
 *
 * @param {Stream} sources
 * @param {Object} opt
 * @returns {Function}
 */
function collectFilesToInject (sources, opt) {
  var collection = [], done = false, queue = [];

  sources.pipe(es.through(collector(collection, opt), function () {
    done = true;
    while (queue.length) {
      resolve(queue.shift());
    }
  }));

  function resolve (cb) {
    setImmediate(function () {
      cb(collection);
    });
  }

  return function (cb) {
    if (done) {
      resolve(cb);
    } else {
      queue.push(cb);
    }
  };
}

/**
 * Create a file collecting function
 * to be used in es.through
 *
 * @param {Array} collection  Collection to fill with files
 * @param {Object} opt
 * @returns {Function}
 */
function collector (collection, opt) {
  return function (file) {
    if (!file.path) {
      return;
    }

    collection.push(file);
  };
}

/**
 * Get new content for template
 * with all injections made
 *
 * @param {Object} target
 * @param {Array} collection
 * @param {Object} opt
 * @returns {Buffer}
 */
function getNewContent (target, collection, opt) {
  var oldContent = target.contents;
  if (!collection.length) {
    log('Nothing to inject into ' + magenta(target.relative) + '.');
    return oldContent;
  }
  var tags = {};
  var targetExt = extname(target.path);

  var filesPerTags = groupBy(collection, function (file) {
    var ext = extname(file.path);
    var startTag = opt.tags.start(targetExt, ext, opt.starttag);
    var endTag = opt.tags.end(targetExt, ext, opt.endtag);
    var tag = startTag + endTag;
    if (!tags[tag]) {
      tags[tag] = {start: startTag, end: endTag};
    }
    return tag;
  });

  var startAndEndTags = Object.keys(filesPerTags);

  log(cyan(collection.length) + ' files into ' + magenta(target.relative) + '.');

  return new Buffer(startAndEndTags.reduce(function eachInCollection (contents, tag) {
    var files = filesPerTags[tag];
    var startTag = tags[tag].start;
    var endTag = tags[tag].end;

    return contents.replace(
      getInjectorTagsRegExp(startTag, endTag),
      function injector (match, starttag, indent, content, endtag) {
        var starttagArray = opt.removeTags ? [] : [starttag];
        var endtagArray = opt.removeTags ? [] : [endtag];
        return starttagArray
          .concat(files.reduce(function transformFile (lines, file, i) {
            var filepath = getFilepath(file, target, opt);
            var transformedContents = opt.transform(filepath, file, i, files.length, target);
            if (typeof transformedContents !== 'string') {
              return lines;
            }
            return lines.concat(transformedContents.split(/\r?\n/g));
          }, []))
          .concat(endtagArray)
          .join(indent);
      }
    );
  }, String(oldContent)));
}

function getFilepath (sourceFile, targetFile, opt) {
  var base = opt.relative ? path.dirname(targetFile.path) : sourceFile.cwd;

  var filepath = unixify(path.relative(base, sourceFile.path));

  if (opt.ignorePath.length) {
    filepath = removeBasePath(opt.ignorePath, filepath);
  }

  if (opt.addPrefix) {
    filepath = addPrefix(filepath, opt.addPrefix);
  }

  if (opt.addRootSlash) {
    filepath = addRootSlash(filepath);
  } else if(!opt.addPrefix) {
    filepath = removeRootSlash(filepath);
  }

  return filepath;
}

function getTag (tag, ext) {
  return tag.replace('{{ext}}', ext);
}

function getInjectorTagsRegExp (starttag, endtag) {
  return new RegExp('(' + makeWhiteSpaceOptional(escapeForRegExp(starttag)) + ')(\\s*)(\\n|\\r|.)*?(' + makeWhiteSpaceOptional(escapeForRegExp(endtag)) + ')', 'gi');
}

function makeWhiteSpaceOptional (str) {
  return str.replace(/\s+/g, '\\s*');
}

function escapeForRegExp (str) {
  return str.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
}

function unixify (filepath) {
  return filepath.replace(/\\/g, '/');
}
function addRootSlash (filepath) {
  return filepath.replace(/^\/*([^\/])/, '/$1');
}
function removeRootSlash (filepath) {
  return filepath.replace(/^\/+/, '');
}
function addPrefix (filepath, prefix) {
  return  prefix + addRootSlash(filepath);
}

function removeBasePath (basedir, filepath) {
  return toArray(basedir).reduce(function (path, remove) {
    if (path[0] === '/' && remove[0] !== '/') {
      remove = '/' + remove;
    }
    if (path[0] !== '/' && remove[0] === '/') {
      path = '/' + path;
    }
    if (remove && path.indexOf(remove) === 0) {
      return path.slice(remove.length);
    }
    return path;
  }, filepath);
}

function toArray (arr) {
  if (!Array.isArray(arr)) {
    return arr ? [arr] : [];
  }
  return arr;
}

function groupBy (arr, cb) {
  var result = {};
  for (var i = 0; i < arr.length; i++) {
    var key = cb(arr[i]);
    if (!result[key]) {
      result[key] = [];
    }
    result[key].push(arr[i]);
  }
  return result;
}

function log (message) {
  gutil.log(magenta(PLUGIN_NAME), message);
}

function warn (message) {
  log(red('WARNING') + ' ' + message);
}

function error (message) {
  return new PluginError(PLUGIN_NAME, message);
}
