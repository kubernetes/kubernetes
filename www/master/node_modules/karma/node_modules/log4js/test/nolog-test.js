"use strict";
var vows = require('vows')
, assert = require('assert')
, util   = require('util')
, EE     = require('events').EventEmitter
, levels = require('../lib/levels');

function MockLogger() {

  var that = this;
  this.messages = [];

  this.log = function(level, message, exception) {
    that.messages.push({ level: level, message: message });
  };

  this.isLevelEnabled = function(level) {
    return level.isGreaterThanOrEqualTo(that.level);
  };

  this.level = levels.TRACE;

}

function MockRequest(remoteAddr, method, originalUrl) {
  
  this.socket = { remoteAddress: remoteAddr };
  this.originalUrl = originalUrl;
  this.method = method;
  this.httpVersionMajor = '5';
  this.httpVersionMinor = '0';
  this.headers = {};
}

function MockResponse(statusCode) {
  var r = this;
  this.statusCode = statusCode;

  this.end = function(chunk, encoding) {
      r.emit('finish');
  };
}
util.inherits(MockResponse, EE);

vows.describe('log4js connect logger').addBatch({
  'getConnectLoggerModule': {
    topic: function() {
      var clm = require('../lib/connect-logger');
      return clm;
    },

    'should return a "connect logger" factory' : function(clm) {
      assert.isObject(clm);
    },

    'nolog String' : {
      topic: function(clm) {
        var ml = new MockLogger();
        var cl = clm.connectLogger(ml, { nolog: "\\.gif" });
        return {cl: cl, ml: ml};
      },

      'check unmatch url request': {
        topic: function(d){
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.png'); // not gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() {
              cb(null, d.ml.messages);
          },10);
        }, 
        'check message': function(messages){
          assert.isArray(messages);
          assert.equal(messages.length, 1);
          assert.ok(levels.INFO.isEqualTo(messages[0].level));
          assert.include(messages[0].message, 'GET');
          assert.include(messages[0].message, 'http://url');
          assert.include(messages[0].message, 'my.remote.addr');
          assert.include(messages[0].message, '200');
          messages.pop();
        }
      },

      'check match url request': {
        topic: function(d) {
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.gif'); // gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() {
              cb(null, d.ml.messages);
          },10);
        }, 
        'check message': function(messages) {
          assert.isArray(messages);
          assert.equal(messages.length, 0);
        }
      }
    },

    'nolog Strings' : {
      topic: function(clm) {
        var ml = new MockLogger();
        var cl = clm.connectLogger(ml, {nolog: "\\.gif|\\.jpe?g"});
        return {cl: cl, ml: ml};
      },
      
      'check unmatch url request (png)': {
        topic: function(d){
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.png'); // not gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages){
          assert.isArray(messages);
          assert.equal(messages.length, 1);
          assert.ok(levels.INFO.isEqualTo(messages[0].level));
          assert.include(messages[0].message, 'GET');
          assert.include(messages[0].message, 'http://url');
          assert.include(messages[0].message, 'my.remote.addr');
          assert.include(messages[0].message, '200');
          messages.pop();
        }
      },

      'check match url request (gif)': {
        topic: function(d) {
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.gif'); // gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages) {
          assert.isArray(messages);
          assert.equal(messages.length, 0);
        }
      },
      'check match url request (jpeg)': {
        topic: function(d) {
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.jpeg'); // gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages) {
          assert.isArray(messages);
          assert.equal(messages.length, 0);
        }
      }
    },
    'nolog Array<String>' : {
      topic: function(clm) {
        var ml = new MockLogger();
        var cl = clm.connectLogger(ml, {nolog: ["\\.gif", "\\.jpe?g"]});
        return {cl: cl, ml: ml};
      },
      
      'check unmatch url request (png)': {
        topic: function(d){
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.png'); // not gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages){
          assert.isArray(messages);
          assert.equal(messages.length, 1);
          assert.ok(levels.INFO.isEqualTo(messages[0].level));
          assert.include(messages[0].message, 'GET');
          assert.include(messages[0].message, 'http://url');
          assert.include(messages[0].message, 'my.remote.addr');
          assert.include(messages[0].message, '200');
          messages.pop();
        }
      },

      'check match url request (gif)': {
        topic: function(d) {
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.gif'); // gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages) {
          assert.isArray(messages);
          assert.equal(messages.length, 0);
        }
      },

      'check match url request (jpeg)': {
        topic: function(d) {
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.jpeg'); // gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages) {
          assert.isArray(messages);
          assert.equal(messages.length, 0);
        }
      },
    },
    'nolog RegExp' : {
      topic: function(clm) {
        var ml = new MockLogger();
        var cl = clm.connectLogger(ml, {nolog: /\.gif|\.jpe?g/});
        return {cl: cl, ml: ml};
      },

      'check unmatch url request (png)': {
        topic: function(d){
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.png'); // not gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages){
          assert.isArray(messages);
          assert.equal(messages.length, 1);
          assert.ok(levels.INFO.isEqualTo(messages[0].level));
          assert.include(messages[0].message, 'GET');
          assert.include(messages[0].message, 'http://url');
          assert.include(messages[0].message, 'my.remote.addr');
          assert.include(messages[0].message, '200');
          messages.pop();
        }
      },

      'check match url request (gif)': {
        topic: function(d) {
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.gif'); // gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages) {
          assert.isArray(messages);
          assert.equal(messages.length, 0);
        }
      },

      'check match url request (jpeg)': {
        topic: function(d) {
          var req = new MockRequest('my.remote.addr', 'GET', 'http://url/hoge.jpeg'); // gif
          var res = new MockResponse(200);
          var cb  = this.callback;
          d.cl(req, res, function() { });
          res.end('chunk', 'encoding');
          setTimeout(function() { 
            cb(null, d.ml.messages) 
          }, 10);
        }, 
        'check message': function(messages) {
          assert.isArray(messages);
          assert.equal(messages.length, 0);
        }
      }
    }
  }

}).export(module);
