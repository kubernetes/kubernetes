var coffee = require('coffee-script');
var path = require('path');
var createCoffeePreprocessor = function(args, config, logger, helper) {
  config = config || {};

  var log = logger.create('preprocessor.coffee');
  var defaultOptions = {
    bare: true,
    sourceMap: false
  };
  var options = helper.merge(defaultOptions, args.options || {}, config.options || {});

  var transformPath = args.transformPath || config.transformPath || function(filepath) {
    return filepath.replace(/\.coffee$/, '.js');
  };

  return function(content, file, done) {
    var result = null;
    var map;
    var datauri;

    log.debug('Processing "%s".', file.originalPath);
    file.path = transformPath(file.originalPath);

    // Clone the options because coffee.compile mutates them
    var opts = helper._.clone(options)

    try {
      result = coffee.compile(content, opts);
    } catch (e) {
      log.error('%s\n  at %s:%d', e.message, file.originalPath, e.location.first_line);
      return;
    }

    if (result.v3SourceMap) {
      map = JSON.parse(result.v3SourceMap)
      map.sources[0] = path.basename(file.originalPath)
      map.sourcesContent = [content]
      map.file = path.basename(file.path)
      file.sourceMap = map;
      datauri = 'data:application/json;charset=utf-8;base64,' + new Buffer(JSON.stringify(map)).toString('base64')
      done(result.js + '\n//@ sourceMappingURL=' + datauri + '\n');
    } else {
      done(result.js || result)
    }
  };
};

createCoffeePreprocessor.$inject = ['args', 'config.coffeePreprocessor', 'logger', 'helper'];

// PUBLISH DI MODULE
module.exports = {
  'preprocessor:coffee': ['factory', createCoffeePreprocessor]
};
