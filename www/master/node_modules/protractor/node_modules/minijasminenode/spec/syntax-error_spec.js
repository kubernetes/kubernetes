var minijasminelib = require('../lib/index');

describe('syntax-error', function() {
  var env;
  beforeEach(function() {
    env = new jasmine.Env();
    // Hide the failure result on the console
    env.addReporter = function() {};
  });

  it('should report a failure when a syntax error happens', function() {
    minijasminelib.executeSpecs({
      specs: ['spec/syntax_error.js'],
      jasmineEnv: env
    });
    expect(env.currentRunner().results().failedCount).toEqual(1);
    var firstResult = env.currentRunner().results().getItems()[0];
    var firstSpecResult = firstResult.getItems()[0].getItems()[0];
    expect(firstSpecResult.message).toMatch('SyntaxError');
  });
});
