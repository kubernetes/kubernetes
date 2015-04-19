/**
 * Utility functions for command line output logging from Protractor.
 * May be used in different processes, since the launcher spawns
 * child processes for each runner.
 *
 * All logging directly from Protractor, its driver providers, or runners,
 * should go through this file so that it can be customized.
 */

var troubleshoot = false;

var set = function(config) {
  troubleshoot = config.troubleshoot;
};

var print = function(msg) {
  process.stdout.write(msg);
};

var puts = function() {
  console.log.apply(console, arguments);
};

var debug = function(msg) {
  if (troubleshoot) {
    puts('DEBUG - ' + msg);
  }
};

var warn = function(msg) {
  puts('WARNING - ' + msg);
};

var error = function(msg) {
  puts('ERROR - ' + msg);
};

exports.set = set;
exports.print = print;
exports.puts = puts;
exports.debug = debug;
exports.warn = warn;
exports.error = error;
