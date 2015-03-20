'use strict';

// Karma configuration
module.exports = function(config) {
  var _ = require('lodash'),
    basePath = '.',
    assets = require(basePath + '/config/assets.json');

  config.set({

    // base path, that will be used to resolve files and exclude
    basePath: basePath,

    // frameworks to use
    frameworks: ['jasmine'],

    // list of files / patterns to load in the browser
    files: _.flatten(_.values(assets.core.js)).concat([
      'packages/*/public/*.js',
      'packages/*/public/*/*.js'
    ]),

    // list of files to exclude
    exclude: [],

    // test results reporter to use
    // possible values: 'dots', 'progress', 'junit', 'growl', 'coverage'
    reporters: ['progress', 'coverage'],

    // coverage
    preprocessors: {
      // source files that you want to generate coverage for
      // do not include tests or libraries
      // (these files will be instrumented by Istanbul)
      'packages/*/public/controllers/*.js': ['coverage'],
      'packages/*/public/services/*.js': ['coverage']
    },

    coverageReporter: {
      type: 'html',
      dir: 'test/coverage/'
    },

    // web server port
    port: 9876,
    // Look for server on port 3001 (invoked by mocha) - via @brownman
    proxies: {
      '/': 'http://localhost:3001/'
    },
    
    // enable / disable colors in the output (reporters and logs)
    colors: true,

    // level of logging
    // possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
    logLevel: config.LOG_DEBUG,

    // enable / disable watching file and executing tests whenever any file changes
    autoWatch: true,

    // Start these browsers, currently available:
    // - Chrome
    // - ChromeCanary
    // - Firefox
    // - Opera
    // - Safari (only Mac)
    // - PhantomJS
    // - IE (only Windows)
    browsers: ['PhantomJS'],

    // If browser does not capture in given timeout [ms], kill it
    captureTimeout: 60000,

    // Continuous Integration mode
    // if true, it capture browsers, run tests and exit
    singleRun: true
  });
};
