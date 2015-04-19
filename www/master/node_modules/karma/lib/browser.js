var helper = require('./helper');
var events = require('./events');
var logger = require('./logger');


var Result = function() {
  var startTime = Date.now();

  this.total = this.skipped = this.failed = this.success = 0;
  this.netTime = this.totalTime = 0;
  this.disconnected = this.error = false;

  this.totalTimeEnd = function() {
    this.totalTime = Date.now() - startTime;
  };
};


// The browser is ready to execute tests.
var READY = 1;

// The browser is executing the tests/
var EXECUTING = 2;

// The browser is not executing, but temporarily disconnected (waiting for reconnecting).
var READY_DISCONNECTED = 3;

// The browser is executing the tests, but temporarily disconnect (waiting for reconnecting).
var EXECUTING_DISCONNECTED = 4;

// The browser got permanently disconnected (being removed from the collection and destroyed).
var DISCONNECTED = 5;


var Browser = function(id, fullName, /* capturedBrowsers */ collection, emitter, socket, timer,
    /* config.browserDisconnectTimeout */ disconnectDelay) {

  var name = helper.browserFullNameToShort(fullName);
  var log = logger.create(name);

  this.id = id;
  this.fullName = fullName;
  this.name = name;
  this.state = READY;
  this.lastResult = new Result();

  this.init = function() {
    collection.add(this);

    events.bindAll(this, socket);

    log.info('Connected on socket %s', socket.id);

    // TODO(vojta): remove launchId,
    // it's here just for WebStorm B-C.
    this.launchId = this.id;
    this.id = socket.id;

    // TODO(vojta): move to collection
    emitter.emit('browsers_change', collection);

    emitter.emit('browser_register', this);
  };

  this.isReady = function() {
    return this.state === READY;
  };

  this.toString = function() {
    return this.name;
  };

  this.onError = function(error) {
    if (this.isReady()) {
      return;
    }

    this.lastResult.error = true;
    emitter.emit('browser_error', this, error);
  };

  this.onInfo = function(info) {
    if (this.isReady()) {
      return;
    }

    // TODO(vojta): remove
    if (helper.isDefined(info.dump)) {
      emitter.emit('browser_log', this, info.dump, 'dump');
    }

    if (helper.isDefined(info.log)) {
      emitter.emit('browser_log', this, info.log, info.type);
    }

    if (helper.isDefined(info.total)) {
      this.lastResult.total = info.total;
    }
  };

  this.onComplete = function(result) {
    if (this.isReady()) {
      return;
    }

    this.state = READY;
    this.lastResult.totalTimeEnd();

    if (!this.lastResult.success) {
      this.lastResult.error = true;
    }

    emitter.emit('browsers_change', collection);
    emitter.emit('browser_complete', this, result);
  };

  var self = this;
  var disconnect = function() {
    self.state = DISCONNECTED;
    log.warn('Disconnected');
    collection.remove(self);
  };

  var pendingDisconnect;
  this.onDisconnect = function() {
    if (this.state === READY) {
      disconnect();
    } else if (this.state === EXECUTING) {
      log.debug('Disconnected during run, waiting for reconnecting.');
      this.state = EXECUTING_DISCONNECTED;

      pendingDisconnect = timer.setTimeout(function() {
        self.lastResult.totalTimeEnd();
        self.lastResult.disconnected = true;
        disconnect();
        emitter.emit('browser_complete', self);
      }, disconnectDelay);
    }
  };

  this.onReconnect = function(newSocket) {
    if (this.state === EXECUTING_DISCONNECTED) {
      this.state = EXECUTING;
      log.debug('Reconnected.');
    } else if (this.state === EXECUTING || this.state === READY) {
      log.debug('New connection, forgetting the old one.');
      // TODO(vojta): this should only remove this browser.onDisconnect listener
      socket.removeAllListeners('disconnect');
    }

    socket = newSocket;
    events.bindAll(this, newSocket);
    if (pendingDisconnect) {
      timer.clearTimeout(pendingDisconnect);
    }
  };

  this.onResult = function(result) {
    if (result.length) {
      return result.forEach(this.onResult, this);
    }

    // ignore - probably results from last run (after server disconnecting)
    if (this.isReady()) {
      return;
    }

    if (result.skipped) {
      this.lastResult.skipped++;
    } else if (result.success) {
      this.lastResult.success++;
    } else {
      this.lastResult.failed++;
    }

    this.lastResult.netTime += result.time;
    emitter.emit('spec_complete', this, result);
  };

  this.serialize = function() {
    return {
      id: this.id,
      name: this.name,
      isReady: this.state === READY
    };
  };
};

Browser.STATE_READY = READY;
Browser.STATE_EXECUTING = EXECUTING;
Browser.STATE_READY_DISCONNECTED = READY_DISCONNECTED;
Browser.STATE_EXECUTING_DISCONNECTED = EXECUTING_DISCONNECTED;
Browser.STATE_DISCONNECTED = DISCONNECTED;


var Collection = function(emitter, browsers) {
  browsers = browsers || [];

  this.add = function(browser) {
    browsers.push(browser);
    emitter.emit('browsers_change', this);
  };

  this.remove = function(browser) {
    var index = browsers.indexOf(browser);

    if (index === -1) {
      return false;
    }

    browsers.splice(index, 1);
    emitter.emit('browsers_change', this);

    return true;
  };

  this.getById = function(browserId) {
    for (var i = 0; i < browsers.length; i++) {
      // TODO(vojta): use id, once we fix WebStorm plugin
      if (browsers[i].launchId === browserId) {
        return browsers[i];
      }
    }

    return null;
  };

  this.setAllToExecuting = function() {
    browsers.forEach(function(browser) {
      browser.state = EXECUTING;
    });

    emitter.emit('browsers_change', this);
  };

  this.areAllReady = function(nonReadyList) {
    nonReadyList = nonReadyList || [];

    browsers.forEach(function(browser) {
      if (!browser.isReady()) {
        nonReadyList.push(browser);
      }
    });

    return nonReadyList.length === 0;
  };

  this.serialize = function() {
    return browsers.map(function(browser) {
      return browser.serialize();
    });
  };

  this.getResults = function() {
    var results = browsers.reduce(function(previous, current) {
      previous.success += current.lastResult.success;
      previous.failed += current.lastResult.failed;
      previous.error = previous.error || current.lastResult.error;
      previous.disconnected = previous.disconnected || current.lastResult.disconnected;
      return previous;
    }, {success: 0, failed: 0, error: false, disconnected: false, exitCode: 0});

    // compute exit status code
    results.exitCode = results.failed || results.error || results.disconnected ? 1 : 0;

    return results;
  };

  this.clearResults = function() {
    browsers.forEach(function(browser) {
      browser.lastResult = new Result();
    });
  };

  this.clone = function() {
    return new Collection(emitter, browsers.slice());
  };

  // Array APIs
  this.map = function(callback, context) {
    return browsers.map(callback, context);
  };

  this.forEach = function(callback, context) {
    return browsers.forEach(callback, context);
  };

  // this.length
  Object.defineProperty(this, 'length', {
    get: function() {
      return browsers.length;
    }
  });
};
Collection.$inject = ['emitter'];

exports.Result = Result;
exports.Browser = Browser;
exports.Collection = Collection;
