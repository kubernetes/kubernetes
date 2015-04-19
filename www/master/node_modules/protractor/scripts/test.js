#!/usr/bin/env node

var Executor = require('./test/test_util').Executor;

var passingTests = [
  'node lib/cli.js spec/basicConf.js',
  'node lib/cli.js spec/multiConf.js',
  'node lib/cli.js spec/altRootConf.js',
  'node lib/cli.js spec/onCleanUpAsyncReturnValueConf.js',
  'node lib/cli.js spec/onCleanUpNoReturnValueConf.js',
  'node lib/cli.js spec/onCleanUpSyncReturnValueConf.js',
  'node lib/cli.js spec/onPrepareConf.js',
  'node lib/cli.js spec/onPrepareFileConf.js',
  'node lib/cli.js spec/onPreparePromiseConf.js',
  'node lib/cli.js spec/onPreparePromiseFileConf.js',
  'node lib/cli.js spec/mochaConf.js',
  'node lib/cli.js spec/cucumberConf.js',
  'node lib/cli.js spec/withLoginConf.js',
  'node lib/cli.js spec/suitesConf.js --suite okmany',
  'node lib/cli.js spec/suitesConf.js --suite okspec',
  'node lib/cli.js spec/suitesConf.js --suite okmany,okspec',
  'node lib/cli.js spec/pluginsBasicConf.js',
  'node lib/cli.js spec/pluginsFullConf.js',
  'node lib/cli.js spec/interactionConf.js',
  'node lib/cli.js spec/directConnectConf.js',
  'node lib/cli.js spec/restartBrowserBetweenTestsConf.js',
  'node lib/cli.js spec/getCapabilitiesConf.js',
  'node lib/cli.js spec/controlLockConf.js',
  'node lib/cli.js spec/customFramework.js',
  'node node_modules/.bin/jasmine JASMINE_CONFIG_PATH=scripts/unit_test.json'
];

// Plugins
passingTests.push('node node_modules/minijasminenode/bin/minijn ' +
    'plugins/timeline/spec/unit.js');
passingTests.push(
    'node lib/cli.js plugins/timeline/spec/conf.js',
    'node lib/cli.js plugins/ngHint/spec/successConfig.js',
    'node lib/cli.js plugins/accessibility/spec/successConfig.js');

var executor = new Executor();

passingTests.forEach(function(passing_test) {
  executor.addCommandlineTest(passing_test)
      .assertExitCodeOnly();
});

/*************************
 *Below are failure tests*
 *************************/

// assert stacktrace shows line of failure
executor.addCommandlineTest('node lib/cli.js spec/errorTest/singleFailureConf.js')
    .expectExitCode(1)
    .expectErrors({
      stackTrace: 'single_failure_spec1.js:5:32'
    });

// assert timeout works
executor.addCommandlineTest('node lib/cli.js spec/errorTest/timeoutConf.js')
    .expectExitCode(1)
    .expectErrors({
      message: 'Timeout - Async callback was not invoked within timeout ' +
          'specified by jasmine.DEFAULT_TIMEOUT_INTERVAL.'
    })
    .expectTestDuration(0, 100);

executor.addCommandlineTest('node lib/cli.js spec/errorTest/afterLaunchChangesExitCodeConf.js')
    .expectExitCode(11)
    .expectErrors({
      message: 'Expected \'Hiya\' to equal \'INTENTIONALLY INCORRECT\'.'
    });

executor.addCommandlineTest('node lib/cli.js spec/errorTest/multiFailureConf.js')
    .expectExitCode(1)
    .expectErrors([{
      message: 'Expected \'Hiya\' to equal \'INTENTIONALLY INCORRECT\'.',
      stacktrace: 'single_failure_spec1.js:5:32'
    }, {
      message: 'Expected \'Hiya\' to equal \'INTENTIONALLY INCORRECT\'.',
      stacktrace: 'single_failure_spec2.js:5:32'
    }]);

executor.addCommandlineTest('node lib/cli.js spec/errorTest/shardedFailureConf.js')
    .expectExitCode(1)
    .expectErrors([{
      message: 'Expected \'Hiya\' to equal \'INTENTIONALLY INCORRECT\'.',
      stacktrace: 'single_failure_spec1.js:5:32'
    }, {
      message: 'Expected \'Hiya\' to equal \'INTENTIONALLY INCORRECT\'.',
      stacktrace: 'single_failure_spec2.js:5:32'
    }]);

executor.addCommandlineTest('node lib/cli.js spec/errorTest/mochaFailureConf.js')
    .expectExitCode(1)
    .expectErrors([{
      message: 'expected \'My AngularJS App\' to equal \'INTENTIONALLY INCORRECT\'',
      stacktrace: 'mocha_failure_spec.js:11:20'
    }]);

executor.addCommandlineTest('node lib/cli.js spec/errorTest/pluginsFailingConf.js')
    .expectExitCode(1)
    .expectErrors([
      {message: 'Expected true to be false'},
      {message: 'from setup'},
      {message: 'from postTest passing'},
      {message: 'from postTest failing'},
      {message: 'from teardown'}
    ]);

// Check ngHint plugin

executor.addCommandlineTest(
    'node lib/cli.js plugins/ngHint/spec/failureConfig.js')
    .expectExitCode(1)
    .expectErrors([{
      message: 'warning -- ngHint plugin cannot be run as ngHint code was ' +
          'never included into the page'
    }, {
      message: 'warning -- ngHint is included on the page, but is not active ' +
          'because there is no `ng-hint` attribute present'
    }, {
      message: 'warning -- Module "xApp" was created but never loaded.'
    }]);

// Check accessibility plugin

executor.addCommandlineTest(
    'node lib/cli.js plugins/accessibility/spec/failureConfig.js')
    .expectExitCode(1)
    .expectErrors([{
      message: '3 elements failed:'
    },
    {
      message: '1 element failed:'
    }]);

executor.execute();
