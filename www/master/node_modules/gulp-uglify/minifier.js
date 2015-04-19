'use strict';
var through = require('through2');
var deap = require('deap');
var PluginError = require('gulp-util/lib/PluginError');
var applySourceMap = require('vinyl-sourcemaps-apply');
var reSourceMapComment = /\n\/\/# sourceMappingURL=.+?$/;
var pluginName = 'gulp-uglify';

function trycatch(fn, handle) {
  try {
    return fn();
  } catch (e) {
    return handle(e);
  }
}

function setup(opts) {
  var options = deap({}, opts, {
    fromString: true,
    output: {}
  });

  if (options.preserveComments === 'all') {
    options.output.comments = true;
  } else if (options.preserveComments === 'some') {
    // preserve comments with directives or that start with a bang (!)
    options.output.comments = /^!|@preserve|@license|@cc_on/i;
  } else if (typeof options.preserveComments === 'function') {
    options.output.comments = options.preserveComments;
  }

  return options;
}

function createError(file, err) {
  if (typeof err === 'string') {
    return new PluginError(pluginName, file.path + ': ' + err, {
      fileName: file.path,
      showStack: false
    });
  }

  var msg = err.message || err.msg || /* istanbul ignore next */ 'unspecified error';

  return new PluginError(pluginName, file.path + ': ' + msg, {
    fileName: file.path,
    lineNumber: err.line,
    stack: err.stack,
    showStack: false
  });
}

module.exports = function(opts, uglify) {
  function minify(file, encoding, callback) {
    var options = setup(opts || {});

    if (file.isNull()) {
      return callback(null, file);
    }

    if (file.isStream()) {
      return callback(createError(file, 'Streaming not supported'));
    }

    if (file.sourceMap) {
      options.outSourceMap = file.relative;
    }

    var mangled = trycatch(function() {
      var m = uglify.minify(String(file.contents), options);
      m.code = new Buffer(m.code.replace(reSourceMapComment, ''));
      return m;
    }, createError.bind(null, file));

    if (mangled instanceof PluginError) {
      return callback(mangled);
    }

    file.contents = mangled.code;

    if (file.sourceMap) {
      var sourceMap = JSON.parse(mangled.map);
      sourceMap.sources = [file.relative];
      applySourceMap(file, sourceMap);
    }

    callback(null, file);
  }

  return through.obj(minify);
};
