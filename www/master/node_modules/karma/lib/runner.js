var http = require('http');

var constant = require('./constants');
var helper = require('./helper');
var cfg = require('./config');


var parseExitCode = function(buffer, defaultCode) {
  var tailPos = buffer.length - Buffer.byteLength(constant.EXIT_CODE) - 1;

  if (tailPos < 0) {
    return defaultCode;
  }

  // tail buffer which might contain the message
  var tail = buffer.slice(tailPos);
  var tailStr = tail.toString();
  if (tailStr.substr(0, tailStr.length - 1) === constant.EXIT_CODE) {
    tail.fill('\x00');
    return parseInt(tailStr.substr(-1), 10);
  }

  return defaultCode;
};


// TODO(vojta): read config file (port, host, urlRoot)
exports.run = function(config, done) {
  done = helper.isFunction(done) ? done : process.exit;
  config = cfg.parseConfig(config.configFile, config);

  var exitCode = 1;
  var options = {
    hostname: config.hostname,
    path: config.urlRoot + 'run',
    port: config.port,
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  };

  var request = http.request(options, function(response) {
    response.on('data', function(buffer) {
      exitCode = parseExitCode(buffer, exitCode);
      process.stdout.write(buffer);
    });

    response.on('end', function() {
      done(exitCode);
    });
  });

  request.on('error', function(e) {
    if (e.code === 'ECONNREFUSED') {
      console.error('There is no server listening on port %d', options.port);
      done(1);
    } else {
      throw e;
    }
  });

  request.end(JSON.stringify({
    args: config.clientArgs,
    removedFiles: config.removedFiles,
    changedFiles: config.changedFiles,
    addedFiles: config.addedFiles,
    refresh: config.refresh
  }));
};
