(function(window) {

var formatFailedStep = function(step) {

  var stack = step.trace.stack;
  var message = step.message;
  if (stack) {
    // remove the trailing dot
    var firstLine = stack.substring(0, stack.indexOf('\n') - 1);
    if (message && message.indexOf(firstLine) === -1) {
      stack = message + '\n' + stack;
    }

    // remove jasmine stack entries
    return stack.replace(/\n.+jasmine\.js\?\w*\:.+(?=(\n|$))/g, '');
  }

  return message;
};

var indexOf = function(collection, item) {
  if (collection.indexOf) {
    return collection.indexOf(item);
  }

  for (var i = 0, ii = collection.length; i < ii; i++) {
    if (collection[i] === item) {
      return i;
    }
  }

  return -1;
};


// TODO(vojta): Karma might provide this
var getCurrentTransport = function() {
  var parentWindow = window.opener || window.parent;
  var location = parentWindow.location;
  var hostname = 'http://' + location.host;

  if (!location.port) {
    hostname += ':80';
  }

  // Probably running in debug.html (there's no socket.io),
  // or in debug mode with socket.io but no socket on this host.
  if (!parentWindow.io || !parentWindow.io.sockets[hostname]) {
    return null;
  }

  return parentWindow.io.sockets[hostname].transport.name;
};


/**
 * Very simple reporter for jasmine
 */
var KarmaReporter = function(tc) {

  var getAllSpecNames = function(topLevelSuites) {
    var specNames = {};

    var processSuite = function(suite, pointer) {
      var childSuite;
      var childPointer;

      for (var i = 0; i < suite.suites_.length; i++) {
        childSuite = suite.suites_[i];
        childPointer = pointer[childSuite.description] = {};
        processSuite(childSuite, childPointer);
      }

      pointer._ = [];
      for (var j = 0; j < suite.specs_.length; j++) {
        pointer._.push(suite.specs_[j].description);
      }
    };

    var suite;
    var pointer;
    for (var k = 0; k < topLevelSuites.length; k++) {
      suite = topLevelSuites[k];
      pointer = specNames[suite.description] = {};
      processSuite(suite, pointer);
    }

    return specNames;
  };

  this.reportRunnerStarting = function(runner) {
    var transport = getCurrentTransport();
    var specNames = null;

    // This structure can be pretty huge and it blows up socket.io connection, when polling.
    // https://github.com/LearnBoost/socket.io-client/issues/569
    if (transport === 'websocket' || transport === 'flashsocket') {
      specNames = getAllSpecNames(runner.topLevelSuites());
    }

    tc.info({total: runner.specs().length, specs: specNames});
  };

  this.reportRunnerResults = function(runner) {
    tc.complete({
      coverage: window.__coverage__
    });
  };

  this.reportSuiteResults = function(suite) {
    // memory clean up
    suite.after_ = null;
    suite.before_ = null;
    suite.queue = null;
  };

  this.reportSpecStarting = function(spec) {
    spec.results_.time = new Date().getTime();
  };

  this.reportSpecResults = function(spec) {
    var result = {
      id: spec.id,
      description: spec.description,
      suite: [],
      success: spec.results_.failedCount === 0,
      skipped: spec.results_.skipped,
      time: spec.results_.skipped ? 0 : new Date().getTime() - spec.results_.time,
      log: []
    };

    var suitePointer = spec.suite;
    while (suitePointer) {
      result.suite.unshift(suitePointer.description);
      suitePointer = suitePointer.parentSuite;
    }

    if (!result.success) {
      var steps = spec.results_.items_;
      for (var i = 0; i < steps.length; i++) {
        if (!steps[i].passed_) {
          result.log.push(formatFailedStep(steps[i]));
        }
      }
    }

    tc.result(result);

    // memory clean up
    spec.results_ = null;
    spec.spies_ = null;
    spec.queue = null;
  };

  this.log = function() {};
};


var createStartFn = function(tc, jasmineEnvPassedIn) {
  return function(config) {
    // we pass jasmineEnv during testing
    // in production we ask for it lazily, so that adapter can be loaded even before jasmine
    var jasmineEnv = jasmineEnvPassedIn || window.jasmine.getEnv();

    jasmineEnv.addReporter(new KarmaReporter(tc));
    jasmineEnv.execute();
  };
};


window.__karma__.start = createStartFn(window.__karma__);

})(window);
