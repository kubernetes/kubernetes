(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
(function() {
  var Connector, PROTOCOL_6, PROTOCOL_7, Parser, Version, _ref;

  _ref = require('./protocol'), Parser = _ref.Parser, PROTOCOL_6 = _ref.PROTOCOL_6, PROTOCOL_7 = _ref.PROTOCOL_7;

  Version = '2.2.2';

  exports.Connector = Connector = (function() {
    function Connector(options, WebSocket, Timer, handlers) {
      this.options = options;
      this.WebSocket = WebSocket;
      this.Timer = Timer;
      this.handlers = handlers;
      this._uri = "ws" + (this.options.https ? "s" : "") + "://" + this.options.host + ":" + this.options.port + "/livereload";
      this._nextDelay = this.options.mindelay;
      this._connectionDesired = false;
      this.protocol = 0;
      this.protocolParser = new Parser({
        connected: (function(_this) {
          return function(protocol) {
            _this.protocol = protocol;
            _this._handshakeTimeout.stop();
            _this._nextDelay = _this.options.mindelay;
            _this._disconnectionReason = 'broken';
            return _this.handlers.connected(protocol);
          };
        })(this),
        error: (function(_this) {
          return function(e) {
            _this.handlers.error(e);
            return _this._closeOnError();
          };
        })(this),
        message: (function(_this) {
          return function(message) {
            return _this.handlers.message(message);
          };
        })(this)
      });
      this._handshakeTimeout = new Timer((function(_this) {
        return function() {
          if (!_this._isSocketConnected()) {
            return;
          }
          _this._disconnectionReason = 'handshake-timeout';
          return _this.socket.close();
        };
      })(this));
      this._reconnectTimer = new Timer((function(_this) {
        return function() {
          if (!_this._connectionDesired) {
            return;
          }
          return _this.connect();
        };
      })(this));
      this.connect();
    }

    Connector.prototype._isSocketConnected = function() {
      return this.socket && this.socket.readyState === this.WebSocket.OPEN;
    };

    Connector.prototype.connect = function() {
      this._connectionDesired = true;
      if (this._isSocketConnected()) {
        return;
      }
      this._reconnectTimer.stop();
      this._disconnectionReason = 'cannot-connect';
      this.protocolParser.reset();
      this.handlers.connecting();
      this.socket = new this.WebSocket(this._uri);
      this.socket.onopen = (function(_this) {
        return function(e) {
          return _this._onopen(e);
        };
      })(this);
      this.socket.onclose = (function(_this) {
        return function(e) {
          return _this._onclose(e);
        };
      })(this);
      this.socket.onmessage = (function(_this) {
        return function(e) {
          return _this._onmessage(e);
        };
      })(this);
      return this.socket.onerror = (function(_this) {
        return function(e) {
          return _this._onerror(e);
        };
      })(this);
    };

    Connector.prototype.disconnect = function() {
      this._connectionDesired = false;
      this._reconnectTimer.stop();
      if (!this._isSocketConnected()) {
        return;
      }
      this._disconnectionReason = 'manual';
      return this.socket.close();
    };

    Connector.prototype._scheduleReconnection = function() {
      if (!this._connectionDesired) {
        return;
      }
      if (!this._reconnectTimer.running) {
        this._reconnectTimer.start(this._nextDelay);
        return this._nextDelay = Math.min(this.options.maxdelay, this._nextDelay * 2);
      }
    };

    Connector.prototype.sendCommand = function(command) {
      if (this.protocol == null) {
        return;
      }
      return this._sendCommand(command);
    };

    Connector.prototype._sendCommand = function(command) {
      return this.socket.send(JSON.stringify(command));
    };

    Connector.prototype._closeOnError = function() {
      this._handshakeTimeout.stop();
      this._disconnectionReason = 'error';
      return this.socket.close();
    };

    Connector.prototype._onopen = function(e) {
      var hello;
      this.handlers.socketConnected();
      this._disconnectionReason = 'handshake-failed';
      hello = {
        command: 'hello',
        protocols: [PROTOCOL_6, PROTOCOL_7]
      };
      hello.ver = Version;
      if (this.options.ext) {
        hello.ext = this.options.ext;
      }
      if (this.options.extver) {
        hello.extver = this.options.extver;
      }
      if (this.options.snipver) {
        hello.snipver = this.options.snipver;
      }
      this._sendCommand(hello);
      return this._handshakeTimeout.start(this.options.handshake_timeout);
    };

    Connector.prototype._onclose = function(e) {
      this.protocol = 0;
      this.handlers.disconnected(this._disconnectionReason, this._nextDelay);
      return this._scheduleReconnection();
    };

    Connector.prototype._onerror = function(e) {};

    Connector.prototype._onmessage = function(e) {
      return this.protocolParser.process(e.data);
    };

    return Connector;

  })();

}).call(this);

},{"./protocol":6}],2:[function(require,module,exports){
(function() {
  var CustomEvents;

  CustomEvents = {
    bind: function(element, eventName, handler) {
      if (element.addEventListener) {
        return element.addEventListener(eventName, handler, false);
      } else if (element.attachEvent) {
        element[eventName] = 1;
        return element.attachEvent('onpropertychange', function(event) {
          if (event.propertyName === eventName) {
            return handler();
          }
        });
      } else {
        throw new Error("Attempt to attach custom event " + eventName + " to something which isn't a DOMElement");
      }
    },
    fire: function(element, eventName) {
      var event;
      if (element.addEventListener) {
        event = document.createEvent('HTMLEvents');
        event.initEvent(eventName, true, true);
        return document.dispatchEvent(event);
      } else if (element.attachEvent) {
        if (element[eventName]) {
          return element[eventName]++;
        }
      } else {
        throw new Error("Attempt to fire custom event " + eventName + " on something which isn't a DOMElement");
      }
    }
  };

  exports.bind = CustomEvents.bind;

  exports.fire = CustomEvents.fire;

}).call(this);

},{}],3:[function(require,module,exports){
(function() {
  var LessPlugin;

  module.exports = LessPlugin = (function() {
    LessPlugin.identifier = 'less';

    LessPlugin.version = '1.0';

    function LessPlugin(window, host) {
      this.window = window;
      this.host = host;
    }

    LessPlugin.prototype.reload = function(path, options) {
      if (this.window.less && this.window.less.refresh) {
        if (path.match(/\.less$/i)) {
          return this.reloadLess(path);
        }
        if (options.originalPath.match(/\.less$/i)) {
          return this.reloadLess(options.originalPath);
        }
      }
      return false;
    };

    LessPlugin.prototype.reloadLess = function(path) {
      var link, links, _i, _len;
      links = (function() {
        var _i, _len, _ref, _results;
        _ref = document.getElementsByTagName('link');
        _results = [];
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          link = _ref[_i];
          if (link.href && link.rel.match(/^stylesheet\/less$/i) || (link.rel.match(/stylesheet/i) && link.type.match(/^text\/(x-)?less$/i))) {
            _results.push(link);
          }
        }
        return _results;
      })();
      if (links.length === 0) {
        return false;
      }
      for (_i = 0, _len = links.length; _i < _len; _i++) {
        link = links[_i];
        link.href = this.host.generateCacheBustUrl(link.href);
      }
      this.host.console.log("LiveReload is asking LESS to recompile all stylesheets");
      this.window.less.refresh(true);
      return true;
    };

    LessPlugin.prototype.analyze = function() {
      return {
        disable: !!(this.window.less && this.window.less.refresh)
      };
    };

    return LessPlugin;

  })();

}).call(this);

},{}],4:[function(require,module,exports){
(function() {
  var Connector, LiveReload, Options, Reloader, Timer,
    __hasProp = {}.hasOwnProperty;

  Connector = require('./connector').Connector;

  Timer = require('./timer').Timer;

  Options = require('./options').Options;

  Reloader = require('./reloader').Reloader;

  exports.LiveReload = LiveReload = (function() {
    function LiveReload(window) {
      var k, v, _ref;
      this.window = window;
      this.listeners = {};
      this.plugins = [];
      this.pluginIdentifiers = {};
      this.console = this.window.console && this.window.console.log && this.window.console.error ? this.window.location.href.match(/LR-verbose/) ? this.window.console : {
        log: function() {},
        error: this.window.console.error.bind(this.window.console)
      } : {
        log: function() {},
        error: function() {}
      };
      if (!(this.WebSocket = this.window.WebSocket || this.window.MozWebSocket)) {
        this.console.error("LiveReload disabled because the browser does not seem to support web sockets");
        return;
      }
      if ('LiveReloadOptions' in window) {
        this.options = new Options();
        _ref = window['LiveReloadOptions'];
        for (k in _ref) {
          if (!__hasProp.call(_ref, k)) continue;
          v = _ref[k];
          this.options.set(k, v);
        }
      } else {
        this.options = Options.extract(this.window.document);
        if (!this.options) {
          this.console.error("LiveReload disabled because it could not find its own <SCRIPT> tag");
          return;
        }
      }
      this.reloader = new Reloader(this.window, this.console, Timer);
      this.connector = new Connector(this.options, this.WebSocket, Timer, {
        connecting: (function(_this) {
          return function() {};
        })(this),
        socketConnected: (function(_this) {
          return function() {};
        })(this),
        connected: (function(_this) {
          return function(protocol) {
            var _base;
            if (typeof (_base = _this.listeners).connect === "function") {
              _base.connect();
            }
            _this.log("LiveReload is connected to " + _this.options.host + ":" + _this.options.port + " (protocol v" + protocol + ").");
            return _this.analyze();
          };
        })(this),
        error: (function(_this) {
          return function(e) {
            if (e instanceof ProtocolError) {
              if (typeof console !== "undefined" && console !== null) {
                return console.log("" + e.message + ".");
              }
            } else {
              if (typeof console !== "undefined" && console !== null) {
                return console.log("LiveReload internal error: " + e.message);
              }
            }
          };
        })(this),
        disconnected: (function(_this) {
          return function(reason, nextDelay) {
            var _base;
            if (typeof (_base = _this.listeners).disconnect === "function") {
              _base.disconnect();
            }
            switch (reason) {
              case 'cannot-connect':
                return _this.log("LiveReload cannot connect to " + _this.options.host + ":" + _this.options.port + ", will retry in " + nextDelay + " sec.");
              case 'broken':
                return _this.log("LiveReload disconnected from " + _this.options.host + ":" + _this.options.port + ", reconnecting in " + nextDelay + " sec.");
              case 'handshake-timeout':
                return _this.log("LiveReload cannot connect to " + _this.options.host + ":" + _this.options.port + " (handshake timeout), will retry in " + nextDelay + " sec.");
              case 'handshake-failed':
                return _this.log("LiveReload cannot connect to " + _this.options.host + ":" + _this.options.port + " (handshake failed), will retry in " + nextDelay + " sec.");
              case 'manual':
                break;
              case 'error':
                break;
              default:
                return _this.log("LiveReload disconnected from " + _this.options.host + ":" + _this.options.port + " (" + reason + "), reconnecting in " + nextDelay + " sec.");
            }
          };
        })(this),
        message: (function(_this) {
          return function(message) {
            switch (message.command) {
              case 'reload':
                return _this.performReload(message);
              case 'alert':
                return _this.performAlert(message);
            }
          };
        })(this)
      });
      this.initialized = true;
    }

    LiveReload.prototype.on = function(eventName, handler) {
      return this.listeners[eventName] = handler;
    };

    LiveReload.prototype.log = function(message) {
      return this.console.log("" + message);
    };

    LiveReload.prototype.performReload = function(message) {
      var _ref, _ref1;
      this.log("LiveReload received reload request: " + (JSON.stringify(message, null, 2)));
      return this.reloader.reload(message.path, {
        liveCSS: (_ref = message.liveCSS) != null ? _ref : true,
        liveImg: (_ref1 = message.liveImg) != null ? _ref1 : true,
        originalPath: message.originalPath || '',
        overrideURL: message.overrideURL || '',
        serverURL: "http://" + this.options.host + ":" + this.options.port
      });
    };

    LiveReload.prototype.performAlert = function(message) {
      return alert(message.message);
    };

    LiveReload.prototype.shutDown = function() {
      var _base;
      if (!this.initialized) {
        return;
      }
      this.connector.disconnect();
      this.log("LiveReload disconnected.");
      return typeof (_base = this.listeners).shutdown === "function" ? _base.shutdown() : void 0;
    };

    LiveReload.prototype.hasPlugin = function(identifier) {
      return !!this.pluginIdentifiers[identifier];
    };

    LiveReload.prototype.addPlugin = function(pluginClass) {
      var plugin;
      if (!this.initialized) {
        return;
      }
      if (this.hasPlugin(pluginClass.identifier)) {
        return;
      }
      this.pluginIdentifiers[pluginClass.identifier] = true;
      plugin = new pluginClass(this.window, {
        _livereload: this,
        _reloader: this.reloader,
        _connector: this.connector,
        console: this.console,
        Timer: Timer,
        generateCacheBustUrl: (function(_this) {
          return function(url) {
            return _this.reloader.generateCacheBustUrl(url);
          };
        })(this)
      });
      this.plugins.push(plugin);
      this.reloader.addPlugin(plugin);
    };

    LiveReload.prototype.analyze = function() {
      var plugin, pluginData, pluginsData, _i, _len, _ref;
      if (!this.initialized) {
        return;
      }
      if (!(this.connector.protocol >= 7)) {
        return;
      }
      pluginsData = {};
      _ref = this.plugins;
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        plugin = _ref[_i];
        pluginsData[plugin.constructor.identifier] = pluginData = (typeof plugin.analyze === "function" ? plugin.analyze() : void 0) || {};
        pluginData.version = plugin.constructor.version;
      }
      this.connector.sendCommand({
        command: 'info',
        plugins: pluginsData,
        url: this.window.location.href
      });
    };

    return LiveReload;

  })();

}).call(this);

},{"./connector":1,"./options":5,"./reloader":7,"./timer":9}],5:[function(require,module,exports){
(function() {
  var Options;

  exports.Options = Options = (function() {
    function Options() {
      this.https = false;
      this.host = null;
      this.port = 35729;
      this.snipver = null;
      this.ext = null;
      this.extver = null;
      this.mindelay = 1000;
      this.maxdelay = 60000;
      this.handshake_timeout = 5000;
    }

    Options.prototype.set = function(name, value) {
      if (typeof value === 'undefined') {
        return;
      }
      if (!isNaN(+value)) {
        value = +value;
      }
      return this[name] = value;
    };

    return Options;

  })();

  Options.extract = function(document) {
    var element, keyAndValue, m, mm, options, pair, src, _i, _j, _len, _len1, _ref, _ref1;
    _ref = document.getElementsByTagName('script');
    for (_i = 0, _len = _ref.length; _i < _len; _i++) {
      element = _ref[_i];
      if ((src = element.src) && (m = src.match(/^[^:]+:\/\/(.*)\/z?livereload\.js(?:\?(.*))?$/))) {
        options = new Options();
        options.https = src.indexOf("https") === 0;
        if (mm = m[1].match(/^([^\/:]+)(?::(\d+))?$/)) {
          options.host = mm[1];
          if (mm[2]) {
            options.port = parseInt(mm[2], 10);
          }
        }
        if (m[2]) {
          _ref1 = m[2].split('&');
          for (_j = 0, _len1 = _ref1.length; _j < _len1; _j++) {
            pair = _ref1[_j];
            if ((keyAndValue = pair.split('=')).length > 1) {
              options.set(keyAndValue[0].replace(/-/g, '_'), keyAndValue.slice(1).join('='));
            }
          }
        }
        return options;
      }
    }
    return null;
  };

}).call(this);

},{}],6:[function(require,module,exports){
(function() {
  var PROTOCOL_6, PROTOCOL_7, Parser, ProtocolError,
    __indexOf = [].indexOf || function(item) { for (var i = 0, l = this.length; i < l; i++) { if (i in this && this[i] === item) return i; } return -1; };

  exports.PROTOCOL_6 = PROTOCOL_6 = 'http://livereload.com/protocols/official-6';

  exports.PROTOCOL_7 = PROTOCOL_7 = 'http://livereload.com/protocols/official-7';

  exports.ProtocolError = ProtocolError = (function() {
    function ProtocolError(reason, data) {
      this.message = "LiveReload protocol error (" + reason + ") after receiving data: \"" + data + "\".";
    }

    return ProtocolError;

  })();

  exports.Parser = Parser = (function() {
    function Parser(handlers) {
      this.handlers = handlers;
      this.reset();
    }

    Parser.prototype.reset = function() {
      return this.protocol = null;
    };

    Parser.prototype.process = function(data) {
      var command, e, message, options, _ref;
      try {
        if (this.protocol == null) {
          if (data.match(/^!!ver:([\d.]+)$/)) {
            this.protocol = 6;
          } else if (message = this._parseMessage(data, ['hello'])) {
            if (!message.protocols.length) {
              throw new ProtocolError("no protocols specified in handshake message");
            } else if (__indexOf.call(message.protocols, PROTOCOL_7) >= 0) {
              this.protocol = 7;
            } else if (__indexOf.call(message.protocols, PROTOCOL_6) >= 0) {
              this.protocol = 6;
            } else {
              throw new ProtocolError("no supported protocols found");
            }
          }
          return this.handlers.connected(this.protocol);
        } else if (this.protocol === 6) {
          message = JSON.parse(data);
          if (!message.length) {
            throw new ProtocolError("protocol 6 messages must be arrays");
          }
          command = message[0], options = message[1];
          if (command !== 'refresh') {
            throw new ProtocolError("unknown protocol 6 command");
          }
          return this.handlers.message({
            command: 'reload',
            path: options.path,
            liveCSS: (_ref = options.apply_css_live) != null ? _ref : true
          });
        } else {
          message = this._parseMessage(data, ['reload', 'alert']);
          return this.handlers.message(message);
        }
      } catch (_error) {
        e = _error;
        if (e instanceof ProtocolError) {
          return this.handlers.error(e);
        } else {
          throw e;
        }
      }
    };

    Parser.prototype._parseMessage = function(data, validCommands) {
      var e, message, _ref;
      try {
        message = JSON.parse(data);
      } catch (_error) {
        e = _error;
        throw new ProtocolError('unparsable JSON', data);
      }
      if (!message.command) {
        throw new ProtocolError('missing "command" key', data);
      }
      if (_ref = message.command, __indexOf.call(validCommands, _ref) < 0) {
        throw new ProtocolError("invalid command '" + message.command + "', only valid commands are: " + (validCommands.join(', ')) + ")", data);
      }
      return message;
    };

    return Parser;

  })();

}).call(this);

},{}],7:[function(require,module,exports){
(function() {
  var IMAGE_STYLES, Reloader, numberOfMatchingSegments, pathFromUrl, pathsMatch, pickBestMatch, splitUrl;

  splitUrl = function(url) {
    var hash, index, params;
    if ((index = url.indexOf('#')) >= 0) {
      hash = url.slice(index);
      url = url.slice(0, index);
    } else {
      hash = '';
    }
    if ((index = url.indexOf('?')) >= 0) {
      params = url.slice(index);
      url = url.slice(0, index);
    } else {
      params = '';
    }
    return {
      url: url,
      params: params,
      hash: hash
    };
  };

  pathFromUrl = function(url) {
    var path;
    url = splitUrl(url).url;
    if (url.indexOf('file://') === 0) {
      path = url.replace(/^file:\/\/(localhost)?/, '');
    } else {
      path = url.replace(/^([^:]+:)?\/\/([^:\/]+)(:\d*)?\//, '/');
    }
    return decodeURIComponent(path);
  };

  pickBestMatch = function(path, objects, pathFunc) {
    var bestMatch, object, score, _i, _len;
    bestMatch = {
      score: 0
    };
    for (_i = 0, _len = objects.length; _i < _len; _i++) {
      object = objects[_i];
      score = numberOfMatchingSegments(path, pathFunc(object));
      if (score > bestMatch.score) {
        bestMatch = {
          object: object,
          score: score
        };
      }
    }
    if (bestMatch.score > 0) {
      return bestMatch;
    } else {
      return null;
    }
  };

  numberOfMatchingSegments = function(path1, path2) {
    var comps1, comps2, eqCount, len;
    path1 = path1.replace(/^\/+/, '').toLowerCase();
    path2 = path2.replace(/^\/+/, '').toLowerCase();
    if (path1 === path2) {
      return 10000;
    }
    comps1 = path1.split('/').reverse();
    comps2 = path2.split('/').reverse();
    len = Math.min(comps1.length, comps2.length);
    eqCount = 0;
    while (eqCount < len && comps1[eqCount] === comps2[eqCount]) {
      ++eqCount;
    }
    return eqCount;
  };

  pathsMatch = function(path1, path2) {
    return numberOfMatchingSegments(path1, path2) > 0;
  };

  IMAGE_STYLES = [
    {
      selector: 'background',
      styleNames: ['backgroundImage']
    }, {
      selector: 'border',
      styleNames: ['borderImage', 'webkitBorderImage', 'MozBorderImage']
    }
  ];

  exports.Reloader = Reloader = (function() {
    function Reloader(window, console, Timer) {
      this.window = window;
      this.console = console;
      this.Timer = Timer;
      this.document = this.window.document;
      this.importCacheWaitPeriod = 200;
      this.plugins = [];
    }

    Reloader.prototype.addPlugin = function(plugin) {
      return this.plugins.push(plugin);
    };

    Reloader.prototype.analyze = function(callback) {
      return results;
    };

    Reloader.prototype.reload = function(path, options) {
      var plugin, _base, _i, _len, _ref;
      this.options = options;
      if ((_base = this.options).stylesheetReloadTimeout == null) {
        _base.stylesheetReloadTimeout = 15000;
      }
      _ref = this.plugins;
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        plugin = _ref[_i];
        if (plugin.reload && plugin.reload(path, options)) {
          return;
        }
      }
      if (options.liveCSS) {
        if (path.match(/\.css$/i)) {
          if (this.reloadStylesheet(path)) {
            return;
          }
        }
      }
      if (options.liveImg) {
        if (path.match(/\.(jpe?g|png|gif)$/i)) {
          this.reloadImages(path);
          return;
        }
      }
      return this.reloadPage();
    };

    Reloader.prototype.reloadPage = function() {
      return this.window.document.location.reload();
    };

    Reloader.prototype.reloadImages = function(path) {
      var expando, img, selector, styleNames, styleSheet, _i, _j, _k, _l, _len, _len1, _len2, _len3, _ref, _ref1, _ref2, _ref3, _results;
      expando = this.generateUniqueString();
      _ref = this.document.images;
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        img = _ref[_i];
        if (pathsMatch(path, pathFromUrl(img.src))) {
          img.src = this.generateCacheBustUrl(img.src, expando);
        }
      }
      if (this.document.querySelectorAll) {
        for (_j = 0, _len1 = IMAGE_STYLES.length; _j < _len1; _j++) {
          _ref1 = IMAGE_STYLES[_j], selector = _ref1.selector, styleNames = _ref1.styleNames;
          _ref2 = this.document.querySelectorAll("[style*=" + selector + "]");
          for (_k = 0, _len2 = _ref2.length; _k < _len2; _k++) {
            img = _ref2[_k];
            this.reloadStyleImages(img.style, styleNames, path, expando);
          }
        }
      }
      if (this.document.styleSheets) {
        _ref3 = this.document.styleSheets;
        _results = [];
        for (_l = 0, _len3 = _ref3.length; _l < _len3; _l++) {
          styleSheet = _ref3[_l];
          _results.push(this.reloadStylesheetImages(styleSheet, path, expando));
        }
        return _results;
      }
    };

    Reloader.prototype.reloadStylesheetImages = function(styleSheet, path, expando) {
      var e, rule, rules, styleNames, _i, _j, _len, _len1;
      try {
        rules = styleSheet != null ? styleSheet.cssRules : void 0;
      } catch (_error) {
        e = _error;
      }
      if (!rules) {
        return;
      }
      for (_i = 0, _len = rules.length; _i < _len; _i++) {
        rule = rules[_i];
        switch (rule.type) {
          case CSSRule.IMPORT_RULE:
            this.reloadStylesheetImages(rule.styleSheet, path, expando);
            break;
          case CSSRule.STYLE_RULE:
            for (_j = 0, _len1 = IMAGE_STYLES.length; _j < _len1; _j++) {
              styleNames = IMAGE_STYLES[_j].styleNames;
              this.reloadStyleImages(rule.style, styleNames, path, expando);
            }
            break;
          case CSSRule.MEDIA_RULE:
            this.reloadStylesheetImages(rule, path, expando);
        }
      }
    };

    Reloader.prototype.reloadStyleImages = function(style, styleNames, path, expando) {
      var newValue, styleName, value, _i, _len;
      for (_i = 0, _len = styleNames.length; _i < _len; _i++) {
        styleName = styleNames[_i];
        value = style[styleName];
        if (typeof value === 'string') {
          newValue = value.replace(/\burl\s*\(([^)]*)\)/, (function(_this) {
            return function(match, src) {
              if (pathsMatch(path, pathFromUrl(src))) {
                return "url(" + (_this.generateCacheBustUrl(src, expando)) + ")";
              } else {
                return match;
              }
            };
          })(this));
          if (newValue !== value) {
            style[styleName] = newValue;
          }
        }
      }
    };

    Reloader.prototype.reloadStylesheet = function(path) {
      var imported, link, links, match, style, _i, _j, _k, _l, _len, _len1, _len2, _len3, _ref, _ref1;
      links = (function() {
        var _i, _len, _ref, _results;
        _ref = this.document.getElementsByTagName('link');
        _results = [];
        for (_i = 0, _len = _ref.length; _i < _len; _i++) {
          link = _ref[_i];
          if (link.rel.match(/^stylesheet$/i) && !link.__LiveReload_pendingRemoval) {
            _results.push(link);
          }
        }
        return _results;
      }).call(this);
      imported = [];
      _ref = this.document.getElementsByTagName('style');
      for (_i = 0, _len = _ref.length; _i < _len; _i++) {
        style = _ref[_i];
        if (style.sheet) {
          this.collectImportedStylesheets(style, style.sheet, imported);
        }
      }
      for (_j = 0, _len1 = links.length; _j < _len1; _j++) {
        link = links[_j];
        this.collectImportedStylesheets(link, link.sheet, imported);
      }
      if (this.window.StyleFix && this.document.querySelectorAll) {
        _ref1 = this.document.querySelectorAll('style[data-href]');
        for (_k = 0, _len2 = _ref1.length; _k < _len2; _k++) {
          style = _ref1[_k];
          links.push(style);
        }
      }
      this.console.log("LiveReload found " + links.length + " LINKed stylesheets, " + imported.length + " @imported stylesheets");
      match = pickBestMatch(path, links.concat(imported), (function(_this) {
        return function(l) {
          return pathFromUrl(_this.linkHref(l));
        };
      })(this));
      if (match) {
        if (match.object.rule) {
          this.console.log("LiveReload is reloading imported stylesheet: " + match.object.href);
          this.reattachImportedRule(match.object);
        } else {
          this.console.log("LiveReload is reloading stylesheet: " + (this.linkHref(match.object)));
          this.reattachStylesheetLink(match.object);
        }
      } else {
        this.console.log("LiveReload will reload all stylesheets because path '" + path + "' did not match any specific one");
        for (_l = 0, _len3 = links.length; _l < _len3; _l++) {
          link = links[_l];
          this.reattachStylesheetLink(link);
        }
      }
      return true;
    };

    Reloader.prototype.collectImportedStylesheets = function(link, styleSheet, result) {
      var e, index, rule, rules, _i, _len;
      try {
        rules = styleSheet != null ? styleSheet.cssRules : void 0;
      } catch (_error) {
        e = _error;
      }
      if (rules && rules.length) {
        for (index = _i = 0, _len = rules.length; _i < _len; index = ++_i) {
          rule = rules[index];
          switch (rule.type) {
            case CSSRule.CHARSET_RULE:
              continue;
            case CSSRule.IMPORT_RULE:
              result.push({
                link: link,
                rule: rule,
                index: index,
                href: rule.href
              });
              this.collectImportedStylesheets(link, rule.styleSheet, result);
              break;
            default:
              break;
          }
        }
      }
    };

    Reloader.prototype.waitUntilCssLoads = function(clone, func) {
      var callbackExecuted, executeCallback, poll;
      callbackExecuted = false;
      executeCallback = (function(_this) {
        return function() {
          if (callbackExecuted) {
            return;
          }
          callbackExecuted = true;
          return func();
        };
      })(this);
      clone.onload = (function(_this) {
        return function() {
          _this.console.log("LiveReload: the new stylesheet has finished loading");
          _this.knownToSupportCssOnLoad = true;
          return executeCallback();
        };
      })(this);
      if (!this.knownToSupportCssOnLoad) {
        (poll = (function(_this) {
          return function() {
            if (clone.sheet) {
              _this.console.log("LiveReload is polling until the new CSS finishes loading...");
              return executeCallback();
            } else {
              return _this.Timer.start(50, poll);
            }
          };
        })(this))();
      }
      return this.Timer.start(this.options.stylesheetReloadTimeout, executeCallback);
    };

    Reloader.prototype.linkHref = function(link) {
      return link.href || link.getAttribute('data-href');
    };

    Reloader.prototype.reattachStylesheetLink = function(link) {
      var clone, parent;
      if (link.__LiveReload_pendingRemoval) {
        return;
      }
      link.__LiveReload_pendingRemoval = true;
      if (link.tagName === 'STYLE') {
        clone = this.document.createElement('link');
        clone.rel = 'stylesheet';
        clone.media = link.media;
        clone.disabled = link.disabled;
      } else {
        clone = link.cloneNode(false);
      }
      clone.href = this.generateCacheBustUrl(this.linkHref(link));
      parent = link.parentNode;
      if (parent.lastChild === link) {
        parent.appendChild(clone);
      } else {
        parent.insertBefore(clone, link.nextSibling);
      }
      return this.waitUntilCssLoads(clone, (function(_this) {
        return function() {
          var additionalWaitingTime;
          if (/AppleWebKit/.test(navigator.userAgent)) {
            additionalWaitingTime = 5;
          } else {
            additionalWaitingTime = 200;
          }
          return _this.Timer.start(additionalWaitingTime, function() {
            var _ref;
            if (!link.parentNode) {
              return;
            }
            link.parentNode.removeChild(link);
            clone.onreadystatechange = null;
            return (_ref = _this.window.StyleFix) != null ? _ref.link(clone) : void 0;
          });
        };
      })(this));
    };

    Reloader.prototype.reattachImportedRule = function(_arg) {
      var href, index, link, media, newRule, parent, rule, tempLink;
      rule = _arg.rule, index = _arg.index, link = _arg.link;
      parent = rule.parentStyleSheet;
      href = this.generateCacheBustUrl(rule.href);
      media = rule.media.length ? [].join.call(rule.media, ', ') : '';
      newRule = "@import url(\"" + href + "\") " + media + ";";
      rule.__LiveReload_newHref = href;
      tempLink = this.document.createElement("link");
      tempLink.rel = 'stylesheet';
      tempLink.href = href;
      tempLink.__LiveReload_pendingRemoval = true;
      if (link.parentNode) {
        link.parentNode.insertBefore(tempLink, link);
      }
      return this.Timer.start(this.importCacheWaitPeriod, (function(_this) {
        return function() {
          if (tempLink.parentNode) {
            tempLink.parentNode.removeChild(tempLink);
          }
          if (rule.__LiveReload_newHref !== href) {
            return;
          }
          parent.insertRule(newRule, index);
          parent.deleteRule(index + 1);
          rule = parent.cssRules[index];
          rule.__LiveReload_newHref = href;
          return _this.Timer.start(_this.importCacheWaitPeriod, function() {
            if (rule.__LiveReload_newHref !== href) {
              return;
            }
            parent.insertRule(newRule, index);
            return parent.deleteRule(index + 1);
          });
        };
      })(this));
    };

    Reloader.prototype.generateUniqueString = function() {
      return 'livereload=' + Date.now();
    };

    Reloader.prototype.generateCacheBustUrl = function(url, expando) {
      var hash, oldParams, originalUrl, params, _ref;
      if (expando == null) {
        expando = this.generateUniqueString();
      }
      _ref = splitUrl(url), url = _ref.url, hash = _ref.hash, oldParams = _ref.params;
      if (this.options.overrideURL) {
        if (url.indexOf(this.options.serverURL) < 0) {
          originalUrl = url;
          url = this.options.serverURL + this.options.overrideURL + "?url=" + encodeURIComponent(url);
          this.console.log("LiveReload is overriding source URL " + originalUrl + " with " + url);
        }
      }
      params = oldParams.replace(/(\?|&)livereload=(\d+)/, function(match, sep) {
        return "" + sep + expando;
      });
      if (params === oldParams) {
        if (oldParams.length === 0) {
          params = "?" + expando;
        } else {
          params = "" + oldParams + "&" + expando;
        }
      }
      return url + params + hash;
    };

    return Reloader;

  })();

}).call(this);

},{}],8:[function(require,module,exports){
(function() {
  var CustomEvents, LiveReload, k;

  CustomEvents = require('./customevents');

  LiveReload = window.LiveReload = new (require('./livereload').LiveReload)(window);

  for (k in window) {
    if (k.match(/^LiveReloadPlugin/)) {
      LiveReload.addPlugin(window[k]);
    }
  }

  LiveReload.addPlugin(require('./less'));

  LiveReload.on('shutdown', function() {
    return delete window.LiveReload;
  });

  LiveReload.on('connect', function() {
    return CustomEvents.fire(document, 'LiveReloadConnect');
  });

  LiveReload.on('disconnect', function() {
    return CustomEvents.fire(document, 'LiveReloadDisconnect');
  });

  CustomEvents.bind(document, 'LiveReloadShutDown', function() {
    return LiveReload.shutDown();
  });

}).call(this);

},{"./customevents":2,"./less":3,"./livereload":4}],9:[function(require,module,exports){
(function() {
  var Timer;

  exports.Timer = Timer = (function() {
    function Timer(func) {
      this.func = func;
      this.running = false;
      this.id = null;
      this._handler = (function(_this) {
        return function() {
          _this.running = false;
          _this.id = null;
          return _this.func();
        };
      })(this);
    }

    Timer.prototype.start = function(timeout) {
      if (this.running) {
        clearTimeout(this.id);
      }
      this.id = setTimeout(this._handler, timeout);
      return this.running = true;
    };

    Timer.prototype.stop = function() {
      if (this.running) {
        clearTimeout(this.id);
        this.running = false;
        return this.id = null;
      }
    };

    return Timer;

  })();

  Timer.start = function(timeout, func) {
    return setTimeout(func, timeout);
  };

}).call(this);

},{}]},{},[8]);
