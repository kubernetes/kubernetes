(function(window, document, io) {

var CONTEXT_URL = 'context.html';
var VERSION = '%KARMA_VERSION%';
var KARMA_URL_ROOT = '%KARMA_URL_ROOT%';

// connect socket.io
// https://github.com/LearnBoost/Socket.IO/wiki/Configuring-Socket.IO
var socket = io.connect('http://' + location.host, {
  'reconnection delay': 500,
  'reconnection limit': 2000,
  'resource': KARMA_URL_ROOT.substr(1) + 'socket.io',
  'sync disconnect on unload': true,
  'max reconnection attempts': Infinity
});

var browsersElement = document.getElementById('browsers');
socket.on('info', function(browsers) {
  var items = [], status;
  for (var i = 0; i < browsers.length; i++) {
    status = browsers[i].isReady ? 'idle' : 'executing';
    items.push('<li class="' + status + '">' + browsers[i].name + ' is ' + status + '</li>');
  }
  browsersElement.innerHTML = items.join('\n');
});
socket.on('disconnect', function() {
  browsersElement.innerHTML = '';
});

var titleElement = document.getElementById('title');
var bannerElement = document.getElementById('banner');
var updateStatus = function(status) {
  return function(param) {
    var paramStatus = param ? status.replace('$', param) : status;
    titleElement.innerHTML = 'Karma v' + VERSION + ' - ' + paramStatus;
    bannerElement.className = status === 'connected' ? 'online' : 'offline';
  };
};

socket.on('connect', updateStatus('connected'));
socket.on('disconnect', updateStatus('disconnected'));
socket.on('reconnecting', updateStatus('reconnecting in $ ms...'));
socket.on('reconnect', updateStatus('re-connected'));
socket.on('reconnect_failed', updateStatus('failed to reconnect'));

var instanceOf = function(value, constructorName) {
  return Object.prototype.toString.apply(value) === '[object ' + constructorName + ']';
};

/* jshint unused: false */
var Karma = function(socket, context, navigator, location) {
  var hasError = false;
  var store = {};
  var self = this;

  var resultsBufferLimit = 1;
  var resultsBuffer = [];

  this.VERSION = VERSION;
  this.config = {};

  this.setupContext = function(contextWindow) {
    if (hasError) {
      return;
    }

    var getConsole = function(currentWindow) {
      return currentWindow.console || {
          log: function() {},
          info: function() {},
          warn: function() {},
          error: function() {},
          debug: function() {}
        };
    };

    contextWindow.__karma__ = this;

    // This causes memory leak in Chrome (17.0.963.66)
    contextWindow.onerror = function() {
      return contextWindow.__karma__.error.apply(contextWindow.__karma__, arguments);
    };

    contextWindow.onbeforeunload = function(e, b) {
      if (context.src !== 'about:blank') {
        // TODO(vojta): show what test (with explanation about jasmine.UPDATE_INTERVAL)
        contextWindow.__karma__.error('Some of your tests did a full page reload!');
      }
    };

    // patch the console
    var localConsole = contextWindow.console = getConsole(contextWindow);
    var browserConsoleLog = localConsole.log;
    var logMethods = ['log', 'info', 'warn', 'error', 'debug'];
    var patchConsoleMethod = function(method) {
      var orig = localConsole[method];
      if (!orig) {
        return;
      }
      localConsole[method] = function() {
        self.log(method, arguments);
        return Function.prototype.apply.call(orig, localConsole, arguments);
      };
    };
    for (var i = 0; i < logMethods.length; i++) {
      patchConsoleMethod(logMethods[i]);
    }

    contextWindow.dump = function() {
      self.log('dump', arguments);
    };

    contextWindow.alert = function(msg) {
      self.log('alert', [msg]);
    };
  };

  this.log = function(type, args) {
    var values = [];

    for (var i = 0; i < args.length; i++) {
      values.push(this.stringify(args[i], 3));
    }

    this.info({log: values.join(', '), type: type});
  };

  this.stringify = function(obj, depth) {

    if (depth === 0) {
      return '...';
    }

    if (obj === null) {
      return 'null';
    }

    switch (typeof obj) {
    case 'string':
      return '\'' + obj + '\'';
    case 'undefined':
      return 'undefined';
    case 'function':
      return obj.toString().replace(/\{[\s\S]*\}/, '{ ... }');
    case 'boolean':
      return obj ? 'true' : 'false';
    case 'object':
      var strs = [];
      if (instanceOf(obj, 'Array')) {
        strs.push('[');
        for (var i = 0, ii = obj.length; i < ii; i++) {
          if (i) {
            strs.push(', ');
          }
          strs.push(this.stringify(obj[i], depth - 1));
        }
        strs.push(']');
      } else if (instanceOf(obj, 'Date')) {
        return obj.toString();
      } else if (instanceOf(obj, 'Text')) {
        return obj.nodeValue;
      } else if (instanceOf(obj, 'Comment')) {
        return '<!--' + obj.nodeValue + '-->';
      } else if (obj.outerHTML) {
        return obj.outerHTML;
      } else {
        strs.push(obj.constructor.name);
        strs.push('{');
        var first = true;
        for(var key in obj) {
          if (obj.hasOwnProperty(key)) {
            if (first) { first = false; } else { strs.push(', '); }
            strs.push(key + ': ' + this.stringify(obj[key], depth - 1));
          }
        }
        strs.push('}');
      }
      return strs.join('');
    default:
      return obj;
    }
  };


  var clearContext = function() {
    context.src = 'about:blank';
  };

  // error during js file loading (most likely syntax error)
  // we are not going to execute at all
  this.error = function(msg, url, line) {
    hasError = true;
    socket.emit('error', url ? msg + '\nat ' + url + (line ? ':' + line : '') : msg);
    this.complete();
    return false;
  };

  this.result = function(result) {
    if (resultsBufferLimit === 1) {
      return socket.emit('result', result);
    }

    resultsBuffer.push(result);

    if (resultsBuffer.length === resultsBufferLimit) {
      socket.emit('result', resultsBuffer);
      resultsBuffer = [];
    }
  };

  this.complete = function(result) {
    if (resultsBuffer.length) {
      socket.emit('result', resultsBuffer);
      resultsBuffer = [];
    }

    // give the browser some time to breath, there could be a page reload, but because a bunch of
    // tests could run in the same event loop, we wouldn't notice.
    setTimeout(function() {
      socket.emit('complete', result || {});
      clearContext();
    }, 0);
  };

  this.info = function(info) {
    socket.emit('info', info);
  };

  // all files loaded, let's start the execution
  this.loaded = function() {
    // has error -> cancel
    if (!hasError) {
      this.start(this.config);
    }

    // remove reference to child iframe
    this.start = null;
  };

  this.store = function(key, value) {
    if (typeof value === 'undefined') {
      return store[key];
    }

    if (Object.prototype.toString.apply(value) === '[object Array]') {
      var s = store[key] = [];
      for (var i = 0; i < value.length; i++) {
        s.push(value[i]);
      }
    } else {
      // TODO(vojta): clone objects + deep
      store[key] = value;
    }
  };

  // supposed to be overriden by the context
  // TODO(vojta): support multiple callbacks (queue)
  this.start = this.complete;

  socket.on('execute', function(cfg) {
    // reset hasError and reload the iframe
    hasError = false;
    self.config = cfg;
    context.src = CONTEXT_URL;

    // clear the console before run
    // works only on FF (Safari, Chrome do not allow to clear console from js source)
    if (window.console && window.console.clear) {
      window.console.clear();
    }
  });

  // report browser name, id
  socket.on('connect', function() {
    var transport = socket.socket.transport.name;

    // TODO(vojta): make resultsBufferLimit configurable
    if (transport === 'websocket' || transport === 'flashsocket') {
      resultsBufferLimit = 1;
    } else {
      resultsBufferLimit = 50;
    }

    socket.emit('register', {
      name: navigator.userAgent,
      id: parseInt((location.search.match(/\?id=(.*)/) || [])[1], 10) || null
    });
  });
};


window.karma = new Karma(socket, document.getElementById('context'), window.navigator, window.location);

})(window, document, window.io);
