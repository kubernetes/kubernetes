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
