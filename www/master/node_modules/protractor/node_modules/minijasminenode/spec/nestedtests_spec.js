var minijasminelib = require('../lib/index');

describe('nested calls to executeSpecs', function() {
  var env;
  beforeEach(function() {
    env = new jasmine.Env();
    // Hide the failure result on the console
    env.addReporter = function() {};
  });

  it('should allow a nested call to minijasminelib', function() {
    minijasminelib.executeSpecs({
      specs: ['spec/simplefail.js'],
      jasmineEnv: env
    });
    expect(env.currentRunner().results().failedCount).toEqual(1);
    var firstResult = env.currentRunner().results().getItems()[0];
    var firstSpecResult = firstResult.getItems()[0].getItems()[0];
    expect(firstSpecResult.message).toMatch('Expected true to equal false');
  });
});
