var path = require('path'),
    util = require('util'),
    glob = require('glob');

module.exports = Jasmine;
module.exports.ConsoleReporter = require('./console_reporter');

function Jasmine(options) {
  options = options || {};
  var jasmineCore = options.jasmineCore || require('jasmine-core');
  this.jasmineCorePath = path.join(jasmineCore.files.path, 'jasmine.js');
  this.jasmine = jasmineCore.boot(jasmineCore);
  this.projectBaseDir = options.projectBaseDir || path.resolve();
  this.specFiles = [];
  this.env = this.jasmine.getEnv();
  this.reportersCount = 0;
}

Jasmine.prototype.addSpecFile = function(filePath) {
  this.specFiles.push(filePath);
};

Jasmine.prototype.addReporter = function(reporter) {
  this.env.addReporter(reporter);
  this.reportersCount++;
};

Jasmine.prototype.configureDefaultReporter = function(options) {
  var defaultOnComplete = function(passed) {
    if(passed) {
      process.exit(0);
    }
    else {
      process.exit(1);
    }
  };

  options.timer = options.timer || new this.jasmine.Timer();
  options.print = options.print || function() {
    process.stdout.write(util.format.apply(this, arguments));
  };
  options.showColors = options.hasOwnProperty('showColors') ? options.showColors : true;
  options.onComplete = options.onComplete || defaultOnComplete;
  options.jasmineCorePath = options.jasmineCorePath || this.jasmineCorePath;

  var consoleReporter = new module.exports.ConsoleReporter(options);
  this.addReporter(consoleReporter);
};

Jasmine.prototype.addMatchers = function(matchers) {
  this.jasmine.Expectation.addMatchers(matchers);
};

Jasmine.prototype.loadSpecs = function() {
  this.specFiles.forEach(function(file) {
    require(file);
  });
};

Jasmine.prototype.loadConfigFile = function(configFilePath) {
  var absoluteConfigFilePath = path.resolve(this.projectBaseDir, configFilePath || 'spec/support/jasmine.json');
  var config = require(absoluteConfigFilePath);
  this.loadConfig(config);
};

Jasmine.prototype.loadConfig = function(config) {
  var specDir = config.spec_dir;
  var jasmineRunner = this;
  jasmineRunner.specDir = config.spec_dir;

  if(config.helpers) {
    config.helpers.forEach(function(helperFile) {
      var filePaths = glob.sync(path.join(jasmineRunner.projectBaseDir, jasmineRunner.specDir, helperFile));
      filePaths.forEach(function(filePath) {
        if(jasmineRunner.specFiles.indexOf(filePath) === -1) {
          jasmineRunner.specFiles.push(filePath);
        }
      });
    });
  }

  if(config.spec_files) {
    jasmineRunner.addSpecFiles(config.spec_files);
  }
};

Jasmine.prototype.addSpecFiles = function(files) {
  var jasmineRunner = this;

  files.forEach(function(specFile) {
    var filePaths = glob.sync(path.join(jasmineRunner.projectBaseDir, jasmineRunner.specDir, specFile));
    filePaths.forEach(function(filePath) {
      if(jasmineRunner.specFiles.indexOf(filePath) === -1) {
        jasmineRunner.specFiles.push(filePath);
      }
    });
  });
};

Jasmine.prototype.execute = function(files) {
  if(this.reportersCount === 0) {
    this.configureDefaultReporter({});
  }

  if (files && files.length > 0) {
    this.specDir = '';
    this.specFiles = [];
    this.addSpecFiles(files);
  }

  this.loadSpecs();
  this.env.execute();
};
