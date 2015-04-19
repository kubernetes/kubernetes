var reporters = require(__dirname + "/../lib/reporter");

describe('TerminalReporter', function() {
  beforeEach(function() {
    var config = {}
    this.reporter = new reporters.TerminalReporter(config);
    this.reporter.startedAt = new Date();
  });

  describe("initialize", function() {
    it('initializes print_ from config', function() {
      var config = { print: true };
      this.reporter = new reporters.TerminalReporter(config);
      expect(this.reporter.print_).toBeTruthy();
    });

    it('initializes includeStackTrace_ from config', function () {
        var config = {}
        this.reporter = new reporters.TerminalReporter(config);
        expect(this.reporter.includeStackTrace_).toBeTruthy();
    });

    it('sets the started_ flag to false', function() {
      var config = {}
      this.reporter = new reporters.TerminalReporter(config);
      expect(this.reporter.started_).toBeFalsy();
    });

    it('sets the finished_ flag to false', function() {
      var config = {}
      this.reporter = new reporters.TerminalReporter(config);
      expect(this.reporter.finished_).toBeFalsy();
    });

    it('initializes the failures_ array', function() {
      var config = {}
      this.reporter = new reporters.TerminalReporter(config);
      expect(this.reporter.failures_.length).toEqual(0);
    });

    it('sets the callback_ property to false by default', function() {
      var config = {}
      this.reporter = new reporters.TerminalReporter(config);
      expect(this.reporter.callback_).toEqual(false)
    });

    it('sets the callback_ property to onComplete if supplied', function() {
      var foo = function() { }
      var config = { onComplete: foo }
      this.reporter = new reporters.TerminalReporter(config);
      expect(this.reporter.callback_).toBe(foo)
    });
  });

  describe('when the report runner starts', function() {
    beforeEach(function() {
      this.spy = spyOn(this.reporter, 'printLine_');

      var runner = {
        topLevelSuites: function() {
          var suites = [];
          var suite = { id: 25 };
          suites.push(suite);
          return suites;
        }
      };
      this.reporter.reportRunnerStarting(runner);
    });

    it('sets the started_ field to true', function() {
      expect(this.reporter.started_).toBeTruthy();
    });

    it('sets the startedAt field', function() {
      // instanceof does not work cross-context (such as when run with requirejs)
      var ts = Object.prototype.toString;
      expect(ts.call(this.reporter.startedAt)).toBe(ts.call(new Date()));
    });
  });

  describe('reportRunnerResults', function() {
    beforeEach(function() {
      this.printLineSpy = spyOn(this.reporter, 'printLine_');
    });

    it('generates the report', function() {
      var failuresSpy = spyOn(this.reporter, 'reportFailures_');
      var printRunnerResultsSpy = spyOn(this.reporter, 'printRunnerResults_').
                          andReturn('this is the runner result');

      var callbackSpy = spyOn(this.reporter, 'callback_');

      var runner = {
        results: function() {
          var result = { failedCount: 0 };
          return result;
        },
        specs: function() { return []; }
      };
      this.reporter.startedAt = new Date();

      this.reporter.reportRunnerResults(runner);

      expect(failuresSpy).toHaveBeenCalled();
      expect(this.printLineSpy).toHaveBeenCalled();
      expect(callbackSpy).toHaveBeenCalled();
    });
  });

  describe('reportSpecResults', function() {
    beforeEach(function() {
      this.printSpy = spyOn(this.reporter, 'print_');
      this.spec = {
        id: 1,
        description: 'the spec',
        isSuite: false,
        results: function() {
          var result = {
            passed: function() { return true; }
          }
          return result;
        }
      }
    });

    it('prints a \'.\' for pass', function() {
      this.reporter.reportSpecResults(this.spec);
      expect(this.printSpy).toHaveBeenCalledWith('.');
    });

    it('prints an \'F\' for failure', function() {
      var results = function() {
        var result = {
          passed: function() { return false; },
          items_: []
        }
        return result;
      }
      this.spec.results = results;

      this.reporter.reportSpecResults(this.spec);

      expect(this.printSpy).toHaveBeenCalledWith('F');
    });
  });

  describe('addFailure', function() {
    it('adds message and stackTrace to failures_', function() {
      var spec = {
        suite: {
          getFullName: function() { return 'Suite name' }
        },
        description: 'the spec',
        results: function() {
          var result = {
            items_: function() {
              var theItems = new Array();
              var item = {
                passed_: false,
                message: 'the message',
                trace: {
                  stack: 'the stack'
                }
              }
              theItems.push(item);
              return theItems;
            }.call()
          };
          return result;
        }
      };

      this.reporter.addFailure_(spec);

      var failures = this.reporter.failures_;
      expect(failures.length).toEqual(1);
      var failure = failures[0];
      expect(failure.description).toEqual('Suite name the spec');
      expect(failure.message).toEqual('the message');
      expect(failure.stackTrace).toEqual('the stack');
    });
  });

  describe('prints the runner results', function() {
    beforeEach(function() {
      this.runner = {
        results: function() {
          var _results = {
            totalCount: 23,
            failedCount: 52
          };
          return _results;
        },
        specs: function() {
          var _specs = new Array();
          _specs.push(1);
          return _specs;
        }
      };
    });

    it('uses the specs\'s length, totalCount and failedCount', function() {
      var message = this.reporter.printRunnerResults_(this.runner);
      expect(message).toEqual('1 test, 23 assertions, 52 failures\n');
    });
  });

  describe('reports failures', function() {
    beforeEach(function() {
      this.printSpy = spyOn(this.reporter, 'print_');
      this.printLineSpy = spyOn(this.reporter, 'printLine_');
    });

    it('does not report anything when there are no failures', function() {
      this.reporter.failures_ = new Array();

      this.reporter.reportFailures_();

      expect(this.printLineSpy).not.toHaveBeenCalled();
    });

    it('prints the failures', function() {
      var failure = {
        description: 'the spec',
        message: 'the message',
        stackTrace: 'the stackTrace'
      }

      this.reporter.failures_ = new Array();
      this.reporter.failures_.push(failure);

      this.reporter.reportFailures_();

      var generatedOutput =
                 [ [ '\n' ],
                 [ '\n' ],
                 [ '  1) the spec' ],
                 [ '   Message:' ],
                 [ '     the message' ],
                 [ '   Stacktrace:' ] ];

      expect(this.printLineSpy).toHaveBeenCalled();
      expect(this.printLineSpy.argsForCall).toEqual(generatedOutput);

      expect(this.printSpy).toHaveBeenCalled();
      expect(this.printSpy.argsForCall[0]).toEqual(['Failures:']);
      expect(this.printSpy.argsForCall[1]).toEqual(['     the stackTrace']);
    });

    it('prints the failures without a Stacktrace', function () {
        var config = { includeStackTrace: false };
        this.reporter = new reporters.TerminalReporter(config);
        this.printSpy = spyOn(this.reporter, 'print_');
        this.printLineSpy = spyOn(this.reporter, 'printLine_');

        var failure = {
            description: 'the spec',
            message: 'the message',
            stackTrace: 'the stackTrace'
        }

        this.reporter.failures_ = new Array();
        this.reporter.failures_.push(failure);

        this.reporter.reportFailures_();

        var generatedOutput =
                 [ [ '\n' ],
                 [ '\n' ],
                 [ '  1) the spec' ],
                 [ '   Message:' ],
                 [ '     the message' ] ];

        expect(this.printLineSpy).toHaveBeenCalled();
        expect(this.printLineSpy.argsForCall).toEqual(generatedOutput);

        expect(this.printSpy).toHaveBeenCalled();
        expect(this.printSpy.argsForCall[0]).toEqual(['Failures:']);
        expect(this.printSpy.argsForCall[1]).toBeUndefined();
    });
  });

  describe('when verbose', function() {
    beforeEach(function() {
      var config = {
        isVerbose: true
      };
      this.verboseReporter = new reporters.TerminalReporter(config);
      this.verboseReporter.startedAt = new Date();
      this.verbosePrintSpy = spyOn(this.verboseReporter, 'printLine_');
      this.suite = {
        id: 4,
        description: 'child suite',
        getFullName: function() {return 'child suite'},
        parentSuite: {
          id: 2,
          description: 'parent suite',
          getFullName: function() {return 'parent suite'},
        }
      };
    });

    it('should output the message on success', function() {
      var successSpec = {
        id: 11,
        description: 'the spec',
        suite: this.suite,
        results: function() {
          return {
            passed: function() {
              return true;
            }
          };
        }
      };

      this.verboseReporter.reportSpecResults(successSpec);
      expect(this.verbosePrintSpy).toHaveBeenCalled();
      expect(this.verbosePrintSpy.argsForCall).toEqual([
        ['parent suite'],
        ['  child suite', ],
        ['    the spec - pass'] ]);
    });

    it('should output the message on failure', function() {
      var successSpec = {
        id: 11,
        description: 'the spec',
        suite: this.suite,
        results: function() {
          return {
            items_: [],
            passed: function() {
              return false;
            }
          };
        }
      };

      this.verboseReporter.reportSpecResults(successSpec);
      expect(this.verbosePrintSpy).toHaveBeenCalled();
      expect(this.verbosePrintSpy.argsForCall).toEqual([
        ['parent suite'],
        ['  child suite', ],
        ['    the spec - fail'] ]);
    });
  });
});
