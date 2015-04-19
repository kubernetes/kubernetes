var path = require('path');
var fs = require('graceful-fs');
var crypto = require('crypto');
var mm = require('minimatch');

var log = require('./logger').create('preprocess');

// TODO(vojta): extract get/create temp dir somewhere else (use the same for launchers etc)
var TMP = process.env.TMPDIR || process.env.TMP || process.env.TEMP || '/tmp';

var sha1 = function(data) {
  var hash = crypto.createHash('sha1');
  hash.update(data);
  return hash.digest('hex');
};

// TODO(vojta): instantiate preprocessors at the start to show warnings immediately
var createPreprocessor = function(config, basePath, injector) {
  var patterns = Object.keys(config);
  var alreadyDisplayedWarnings = Object.create(null);

  return function(file, done) {
    var preprocessors = [];
    var nextPreprocessor = function(content) {
      if (!preprocessors.length) {
        file.contentPath = path.normalize(TMP + '/'  + sha1(file.path) + path.extname(file.path));
        return fs.writeFile(file.contentPath, content, function() {
          done();
        });
      }

      preprocessors.shift()(content, file, nextPreprocessor);
    };
    var instantiatePreprocessor = function(name) {
      if (alreadyDisplayedWarnings[name]) {
        return;
      }

      try {
        preprocessors.push(injector.get('preprocessor:' + name));
      } catch (e) {
        if (e.message.indexOf('No provider for "preprocessor:' + name + '"') !== -1) {
          log.warn('Can not load "%s", it is not registered!\n  ' +
                   'Perhaps you are missing some plugin?', name);
        } else {
          log.warn('Can not load "%s"!\n  ' + e.stack, name);
        }

        alreadyDisplayedWarnings[name] = true;
      }
    };

    // collects matching preprocessors
    // TODO(vojta): should we cache this ?
    for (var i = 0; i < patterns.length; i++) {
      if (mm(file.originalPath, patterns[i])) {
        config[patterns[i]].forEach(instantiatePreprocessor);
      }
    }

    if (preprocessors.length) {
      return fs.readFile(file.originalPath, function(err, buffer) {
        nextPreprocessor(buffer.toString());
      });
    }

    return process.nextTick(done);
  };
};
createPreprocessor.$inject = ['config.preprocessors', 'config.basePath', 'injector'];

exports.createPreprocessor = createPreprocessor;
