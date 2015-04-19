/**
 * This serves as the main function for starting a test run that has been
 * requested by the launcher.
 */

var ConfigParser = require('./configParser');
var Runner = require('./runner');
var log = require('./logger');

process.on('message', function(m) {
  switch (m.command) {
    case 'run':
      if (!m.capabilities) {
        throw new Error('Run message missing capabilities');
      }
      // Merge in config file options.
      var configParser = new ConfigParser();
      if (m.configFile) {
        configParser.addFileConfig(m.configFile);
      }
      if (m.additionalConfig) {
        configParser.addConfig(m.additionalConfig);
      }
      var config = configParser.getConfig();
      log.set(config);

      // Grab capabilities to run from launcher.
      config.capabilities = m.capabilities;

      //Get specs to be executed by this runner
      config.specs = m.specs;

      // Launch test run.
      var runner = new Runner(config);

      // Pipe events back to the launcher.
      runner.on('testPass', function() {
        process.send({
          event: 'testPass'
        });
      });
      runner.on('testFail', function() {
        process.send({
          event: 'testFail'
        });
      });
      runner.on('testsDone', function(results) {
        process.send({
          event: 'testsDone',
          results: results
        });
      });

      runner.run().then(function(exitCode) {
        process.exit(exitCode);
      }).catch (function(err) {
        log.puts(err.message);
        process.exit(1);
      });
      break;
    default:
      throw new Error('command ' + m.command + ' is invalid');
  }
});
