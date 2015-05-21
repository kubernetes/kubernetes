module.exports = function(config) {
  config.set({

    basePath: '../',

    files: [
      '../third_party/ui/bower_components/angular/angular.js',
      '../third_party/ui/bower_components/angular-aria/angular-aria.js',
      '../third_party/ui/bower_components/angular-material/angular-material.js',
      '../third_party/ui/bower_components/angular-mocks/angular-mocks.js',
      '../third_party/ui/bower_components/angular-route/angular-route.js',
      '../third_party/ui/bower_components/angularjs-jasmine-matchers/dist/matchers.js',
      '../third_party/ui/bower_components/hammerjs/hammer.js',
      '../third_party/ui/bower_components/lodash/dist/lodash.js',
      'app/assets/js/app.js',
      'app/assets/js/base.js',
      'app/shared/**/*.js',
      'app/vendor/**/*.js',
      'master/shared/**/*.js',
      'master/test/**/*.js',
      'master/components/**/test/**/*.js'
    ],

    autoWatch: true,

    frameworks: ['jasmine'],

    browsers: ['Chrome'],

    plugins: [
      'karma-chrome-launcher',
      'karma-firefox-launcher',
      'karma-jasmine',
      'karma-junit-reporter',
      'karma-story-reporter',
      'karma-phantomjs-launcher'
    ],

    junitReporter: {outputFile: 'test_out/unit.xml', suite: 'unit'}

  });
};
