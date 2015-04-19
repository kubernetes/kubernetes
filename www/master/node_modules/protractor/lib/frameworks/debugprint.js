var util = require('util'),
    q = require('q'),
    log = require('../logger');

/**
 * A debug framework which does not actually run any tests, just spits
 * out the list that would be run.
 *
 * @param {Runner} runner The current Protractor Runner.
 * @param {Array} specs Array of Directory Path Strings.
 * @return {q.Promise} Promise resolved with the test results
 */
exports.run = function(runner, specs) {
  return q.promise(function(resolve) {
    log.puts('Resolved spec files: ' + util.inspect(specs));
    resolve({
      failedCount: 0
    });
  });
};
