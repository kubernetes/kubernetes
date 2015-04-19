var spawn = require('child_process').spawn;
var path = require('path');
var fs = require('fs');
var rimraf = require('rimraf');

var log = require('../logger').create('launcher');
var env = process.env;

var BEING_CAPTURED = 1;
var CAPTURED = 2;
var BEING_KILLED = 3;
var FINISHED = 4;
var BEING_TIMEOUTED = 5;


var BaseBrowser = function(id, emitter, captureTimeout, retryLimit) {
  var self = this;
  var capturingUrl;
  var exitCallbacks = [];

  this.killTimeout = 2000;
  this.id = id;
  this.state = null;
  this._tempDir = path.normalize((env.TMPDIR || env.TMP || env.TEMP || '/tmp') + '/karma-' +
      id.toString());

  this.start = function(url) {
    capturingUrl = url;
    self.state = BEING_CAPTURED;

    try {
      log.debug('Creating temp dir at ' + self._tempDir);
      fs.mkdirSync(self._tempDir);
    } catch (e) {}

    self._start(capturingUrl + '?id=' + self.id);

    if (captureTimeout) {
      setTimeout(self._onTimeout, captureTimeout);
    }
  };


  this._start = function(url) {
    self._execCommand(self._getCommand(), self._getOptions(url));
  };


  this.markCaptured = function() {
    if (self.state === BEING_CAPTURED) {
      self.state = CAPTURED;
    }
  };


  this.isCaptured = function() {
    return self.state === CAPTURED;
  };


  this.kill = function(callback) {
    var exitCallback = callback || function() {};

    log.debug('Killing %s', self.name);
    if (self.state === FINISHED) {
      process.nextTick(exitCallback);
    } else if (self.state === BEING_KILLED) {
      exitCallbacks.push(exitCallback);
    } else {
      self.state = BEING_KILLED;
      self._process.kill();
      exitCallbacks.push(exitCallback);
      setTimeout(self._onKillTimeout, self.killTimeout);
    }
  };


  this._onKillTimeout = function() {
    if (self.state !== BEING_KILLED) {
      return;
    }

    log.warn('%s was not killed in %d ms, sending SIGKILL.', self.name, self.killTimeout);

    self._process.kill('SIGKILL');
  };

  this._onTimeout = function() {
    if (self.state !== BEING_CAPTURED) {
      return;
    }

    log.warn('%s have not captured in %d ms, killing.', self.name, captureTimeout);

    self.state = BEING_TIMEOUTED;
    self._process.kill();
  };


  this.toString = function() {
    return self.name;
  };


  this._getCommand = function() {
    var cmd = path.normalize(env[self.ENV_CMD] || self.DEFAULT_CMD[process.platform]);

    if (!cmd) {
      log.error('No binary for %s browser on your platform.\n\t' +
          'Please, set "%s" env variable.', self.name, self.ENV_CMD);
    }

    return cmd;
  };


  this._execCommand = function(cmd, args) {
    // normalize the cmd, remove quotes (spawn does not like them)
    if (cmd.charAt(0) === cmd.charAt(cmd.length - 1) && '\'`"'.indexOf(cmd.charAt(0)) !== -1) {
      cmd = cmd.substring(1, cmd.length - 1);
      log.warn('The path should not be quoted.\n  Normalized the path to %s', cmd);
    }

    log.debug(cmd + ' ' + args.join(' '));
    self._process = spawn(cmd, args);

    var errorOutput = '';

    self._process.on('close', function(code) {
      self._onProcessExit(code, errorOutput);
    });

    self._process.on('error', function(err) {
      if (err.code === 'ENOENT') {
        retryLimit = 0;
        errorOutput = 'Can not find the binary ' + cmd + '\n\t' +
                      'Please set env variable ' + self.ENV_CMD;
      } else {
        errorOutput += err.toString();
      }
    });

    // Node 0.8 does not emit the error
    if (process.versions.node.indexOf('0.8') === 0) {
      self._process.stderr.on('data', function(data) {
        var msg = data.toString();

        if (msg.indexOf('No such file or directory') !== -1) {
          retryLimit = 0;
          errorOutput = 'Can not find the binary ' + cmd + '\n\t' +
                        'Please set env variable ' + self.ENV_CMD;
        } else {
          errorOutput += msg;
        }
      });
    }
  };


  this._onProcessExit = function(code, errorOutput) {
    log.debug('Process %s exitted with code %d', self.name, code);

    if (self.state === BEING_CAPTURED) {
      log.error('Cannot start %s\n\t%s', self.name, errorOutput);
    }

    if (self.state === CAPTURED) {
      log.error('%s crashed.\n\t%s', self.name, errorOutput);
    }

    retryLimit--;

    if (self.state === BEING_CAPTURED || self.state === BEING_TIMEOUTED) {
      if (retryLimit > 0) {
        return self._cleanUpTmp(function() {
          log.info('Trying to start %s again.', self.name);
          self.start(capturingUrl);
        });
      } else {
        emitter.emit('browser_process_failure', self);
      }
    }

    self.state = FINISHED;
    self._cleanUpTmp(function(err) {
      exitCallbacks.forEach(function(exitCallback) {
        exitCallback(err);
      });
      exitCallbacks = [];
    });
  };


  this._cleanUpTmp = function(done) {
    log.debug('Cleaning temp dir %s', self._tempDir);
    rimraf(self._tempDir, done);
  };


  this._getOptions = function(url) {
    return [url];
  };
};

var baseBrowserDecoratorFactory = function(id, emitter, timeout) {
  return function(self) {
    BaseBrowser.call(self, id, emitter, timeout, 3);
  };
};
baseBrowserDecoratorFactory.$inject = ['id', 'emitter', 'config.captureTimeout'];


// PUBLISH
exports.BaseBrowser = BaseBrowser;
exports.decoratorFactory = baseBrowserDecoratorFactory;
