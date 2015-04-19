var path = require('path');

var logger = require('./logger');
var log = logger.create('config');
var helper = require('./helper');
var constant = require('./constants');

// Coffee is required here to enable config files written in coffee-script.
// It's not directly used in this file.
require('coffee-script');


var Pattern = function(pattern, served, included, watched) {
  this.pattern = pattern;
  this.served = helper.isDefined(served) ? served : true;
  this.included = helper.isDefined(included) ? included : true;
  this.watched = helper.isDefined(watched) ? watched : true;
};

var UrlPattern = function(url) {
  Pattern.call(this, url, false, true, false);
};


var createPatternObject = function(pattern) {
  if (pattern && helper.isString(pattern)) {
    return helper.isUrlAbsolute(pattern) ? new UrlPattern(pattern) : new Pattern(pattern);
  }

  if (helper.isObject(pattern)) {
    if (pattern.pattern && helper.isString(pattern.pattern)) {
      return helper.isUrlAbsolute(pattern.pattern) ?
        new UrlPattern(pattern.pattern) :
        new Pattern(pattern.pattern, pattern.served, pattern.included, pattern.watched);
    }

    log.warn('Invalid pattern %s!\n\tObject is missing "pattern" property.', pattern);
    return new Pattern(null, false, false, false);
  }

  log.warn('Invalid pattern %s!\n\tExpected string or object with "pattern" property.', pattern);
  return new Pattern(null, false, false, false);
};

var normalizeConfig = function(config, configFilePath) {

  var basePathResolve = function(relativePath) {
    if (helper.isUrlAbsolute(relativePath)) {
      return relativePath;
    }

    if (!helper.isDefined(config.basePath) || !helper.isDefined(relativePath)) {
      return '';
    }
    return path.resolve(config.basePath, relativePath);
  };

  var createPatternMapper = function(resolve) {
    return function(objectPattern) {
      objectPattern.pattern = resolve(objectPattern.pattern);

      return objectPattern;
    };
  };

  if (helper.isString(configFilePath)) {
    // resolve basePath
    config.basePath = path.resolve(path.dirname(configFilePath), config.basePath);

    // always ignore the config file itself
    config.exclude.push(configFilePath);
  } else {
    config.basePath = path.resolve(config.basePath || '.');
  }

  config.files = config.files.map(createPatternObject).map(createPatternMapper(basePathResolve));
  config.exclude = config.exclude.map(basePathResolve);
  config.junitReporter.outputFile = basePathResolve(config.junitReporter.outputFile);
  config.coverageReporter.dir = basePathResolve(config.coverageReporter.dir);

  // normalize paths on windows
  config.basePath = helper.normalizeWinPath(config.basePath);
  config.files = config.files.map(createPatternMapper(helper.normalizeWinPath));
  config.exclude = config.exclude.map(helper.normalizeWinPath);
  config.junitReporter.outputFile = helper.normalizeWinPath(config.junitReporter.outputFile);
  config.coverageReporter.dir = helper.normalizeWinPath(config.coverageReporter.dir);

  // normalize urlRoot
  var urlRoot = config.urlRoot;
  if (urlRoot.charAt(0) !== '/') {
    urlRoot = '/' + urlRoot;
  }

  if (urlRoot.charAt(urlRoot.length - 1) !== '/') {
    urlRoot = urlRoot + '/';
  }

  if (urlRoot !== config.urlRoot) {
    log.warn('urlRoot normalized to "%s"', urlRoot);
    config.urlRoot = urlRoot;
  }

  if (config.proxies && config.proxies.hasOwnProperty(config.urlRoot)) {
    log.warn('"%s" is proxied, you should probably change urlRoot to avoid conflicts',
        config.urlRoot);
  }

  if (config.singleRun && config.autoWatch) {
    log.debug('autoWatch set to false, because of singleRun');
    config.autoWatch = false;
  }

  if (helper.isString(config.reporters)) {
    config.reporters = config.reporters.split(',');
  }

  // TODO(vojta): remove
  if (helper.isDefined(config.reporter)) {
    log.warn('"reporter" is deprecated, use "reporters" instead');
  }

  // normalize preprocessors
  var preprocessors = config.preprocessors || {};
  var normalizedPreprocessors = config.preprocessors = Object.create(null);

  Object.keys(preprocessors).forEach(function(pattern) {
    var normalizedPattern = helper.normalizeWinPath(basePathResolve(pattern));

    normalizedPreprocessors[normalizedPattern] = helper.isString(preprocessors[pattern]) ?
        [preprocessors[pattern]] : preprocessors[pattern];
  });


  // define custom launchers/preprocessors/reporters - create an inlined plugin
  var module = Object.create(null);
  var hasSomeInlinedPlugin = false;
  ['launcher', 'preprocessor', 'reporter'].forEach(function(type) {
    var definitions = config['custom' + helper.ucFirst(type) + 's'] || {};

    Object.keys(definitions).forEach(function(name) {
      var definition = definitions[name];

      if (!helper.isObject(definition)) {
        return log.warn('Can not define %s %s. Definition has to be an object.', type, name);
      }

      if (!helper.isString(definition.base)) {
        return log.warn('Can not define %s %s. Missing base %s.', type, name, type);
      }

      var token = type + ':' + definition.base;
      var locals = {
        args: ['value', definition]
      };

      module[type + ':' + name] = ['factory', function(injector) {
        return injector.createChild([locals], [token]).get(token);
      }];
      hasSomeInlinedPlugin = true;
    });
  });

  if (hasSomeInlinedPlugin) {
    config.plugins.push(module);
  }

  return config;
};

// TODO(vojta): remove in 0.11
var CONFIG_SYNTAX_FRAMEWORKS_HELP = '  module.exports = function(config) {\n' +
                                    '    config.set({\n' +
                                    '      // your config\n' +
                                    '      frameworks: ["%s"]\n' +
                                    '    });\n' +
                                    '  };\n';

var CONST_ERR = '%s is not supported anymore. Please use:\n'+CONFIG_SYNTAX_FRAMEWORKS_HELP;
['JASMINE', 'MOCHA', 'QUNIT'].forEach(function(framework) {
  [framework, framework + '_ADAPTER'].forEach(function(name) {
    Object.defineProperty(global, name, {configurable: true, get: function() {
      log.warn(CONST_ERR, name, framework.toLowerCase());
      return __dirname + '/../../karma-' + framework.toLowerCase() + '/lib/' +
          (framework === name ? framework.toLowerCase() : 'adapter') + '.js';
    }});
  });
});

['REQUIRE', 'REQUIRE_ADAPTER'].forEach(function(name) {
  Object.defineProperty(global, name, {configurable: true, get: function() {
    log.warn(CONST_ERR, name, 'requirejs');
    return __dirname + '/../../karma-requirejs/lib/' +
          (name === 'REQUIRE' ? 'require' : 'adapter') + '.js';
  }});
});

['ANGULAR_SCENARIO', 'ANGULAR_SCENARIO_ADAPTER'].forEach(function(name) {
  Object.defineProperty(global, name, {configurable: true, get: function() {
    log.warn(CONST_ERR, name, 'ng-scenario');
    return __dirname + '/../../karma-ng-scenario/lib/' +
          (name === 'ANGULAR_SCENARIO' ? 'angular-scenario' : 'adapter') + '.js';
  }});
});

['LOG_DISABLE', 'LOG_INFO', 'LOG_DEBUG', 'LOG_WARN', 'LOG_ERROR'].forEach(function(name) {
  Object.defineProperty(global, name, {configurable: true, get: function() {
    log.warn('%s is not supported anymore.\n  Please use `config.%s` instead.', name, name);
    return constant[name];
  }});
});

var Config = function() {
  var config = this;

  this.LOG_DISABLE = constant.LOG_DISABLE;
  this.LOG_ERROR = constant.LOG_ERROR;
  this.LOG_WARN = constant.LOG_WARN;
  this.LOG_INFO = constant.LOG_INFO;
  this.LOG_DEBUG = constant.LOG_DEBUG;

  this.set = function(newConfig) {
    Object.keys(newConfig).forEach(function(key) {
      config[key] = newConfig[key];
    });
  };

  // TODO(vojta): remove this in 0.10
  this.configure = function(newConfig) {
    log.warn('config.configure() is deprecated, please use config.set() instead.');
    this.set(newConfig);
  };

  // TODO(vojta): remove this in 0.10
  ['launcher', 'reporter', 'preprocessor'].forEach(function(type) {
    var methodName = 'define' + helper.ucFirst(type);
    var propertyName = 'custom' + helper.ucFirst(type) + 's';

    config[methodName] = function(name, base, options) {
      log.warn('config.%s is deprecated, please use "%s" instead.', methodName, propertyName);

      if (!helper.isString(name)) {
        return log.warn('Can not define %s. Name has to be a string.', type);
      }

      if (!helper.isString(base)) {
        return log.warn('Can not define %s %s. Missing parent %s.', type, name, type);
      }

      if (!helper.isObject(options)) {
        return log.warn('Can not define %s %s. Arguments has to be an object.', type, name);
      }

      config[propertyName] = config[propertyName] || {};
      config[propertyName][name] = options;
      options.base = base;
    };
  });


  // DEFAULT CONFIG
  this.frameworks = [];
  this.port = constant.DEFAULT_PORT;
  this.hostname = constant.DEFAULT_HOSTNAME;
  this.basePath = '';
  this.files = [];
  this.exclude = [];
  this.logLevel = constant.LOG_INFO;
  this.colors = true;
  this.autoWatch = false;
  this.autoWatchBatchDelay = 250;
  this.reporters = ['progress'];
  this.singleRun = false;
  this.browsers = [];
  this.captureTimeout = 60000;
  this.proxies = {};
  this.proxyValidateSSL = true;
  this.preprocessors = {'**/*.coffee': 'coffee', '**/*.html': 'html2js'};
  this.urlRoot = '/';
  this.reportSlowerThan = 0;
  this.loggers = [constant.CONSOLE_APPENDER];
  this.transports = ['websocket', 'flashsocket', 'xhr-polling', 'jsonp-polling'];
  this.plugins = ['karma-*'];
  this.client = {
    args: []
  };
  this.browserDisconnectTimeout = 2000;

  // TODO(vojta): remove in 0.10
  this.junitReporter = {
    outputFile: 'test-results.xml',
    suite: ''
  };

  // TODO(vojta): remove in 0.10
  this.coverageReporter = {
    type: 'html',
    dir: 'coverage'
  };
};

var CONFIG_SYNTAX_HELP = '  module.exports = function(config) {\n' +
                         '    config.set({\n' +
                         '      // your config\n' +
                         '    });\n' +
                         '  };\n';

var parseConfig = function(configFilePath, cliOptions) {
  var configModule;
  if (configFilePath) {
    log.debug('Loading config %s', configFilePath);

    try {
      configModule = require(configFilePath);
    } catch(e) {
      if (e.code === 'MODULE_NOT_FOUND' && e.message.indexOf(configFilePath) !== -1) {
        log.error('File %s does not exist!', configFilePath);
      } else {
        log.error('Invalid config file!\n  ' + e.stack);
      }
      return process.exit(1);
    }
    if (!helper.isFunction(configModule)) {
      log.error('Config file must export a function!\n' + CONFIG_SYNTAX_HELP);
      return process.exit(1);
    }
  } else {
    log.debug('No config file specified.');
    // if no config file path is passed, we define a dummy config module.
    configModule = function() {};
  }

  var config = new Config();

  try {
    configModule(config);
  } catch(e) {
    log.error('Error in config file!\n', e);
    return process.exit(1);
  }

  // merge the config from config file and cliOptions (precendense)
  config.set(cliOptions);

  // configure the logger as soon as we can
  logger.setup(config.logLevel, config.colors, config.loggers);

  return normalizeConfig(config, configFilePath);
};

// PUBLIC API
exports.parseConfig = parseConfig;
exports.Pattern = Pattern;
exports.createPatternObject = createPatternObject;
