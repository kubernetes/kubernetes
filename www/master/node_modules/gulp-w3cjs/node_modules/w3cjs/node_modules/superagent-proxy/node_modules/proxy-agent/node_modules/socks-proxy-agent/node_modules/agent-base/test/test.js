
/**
 * Module dependencies.
 */

var fs = require('fs');
var url = require('url');
var net = require('net');
var tls = require('tls');
var http = require('http');
var https = require('https');
var assert = require('assert');
var Agent = require('../');

describe('Agent', function () {
  describe('"error" event', function () {
    it('should be invoked on `http.ClientRequest` instance if passed to callback function on the first tick', function (done) {
      var agent = new Agent(function (req, opts, fn) {
        fn(new Error('is this caught?'));
      });
      var info = url.parse('http://127.0.0.1/foo');
      info.agent = agent;
      var req = http.get(info);
      req.on('error', function (err) {
        assert.equal('is this caught?', err.message);
        done();
      });
    });
    it('should be invoked on `http.ClientRequest` instance if passed to callback function after the first tick', function (done) {
      var agent = new Agent(function (req, opts, fn) {
        process.nextTick(function () {
          fn(new Error('is this caught?'));
        });
      });
      var info = url.parse('http://127.0.0.1/foo');
      info.agent = agent;
      var req = http.get(info);
      req.on('error', function (err) {
        assert.equal('is this caught?', err.message);
        done();
      });
    });
  });
});

describe('"http" module', function () {
  var server;
  var port;

  // setup test HTTP server
  before(function (done) {
    server = http.createServer();
    server.listen(0, function () {
      port = server.address().port;
      done();
    });
  });

  // shut down test HTTP server
  after(function (done) {
    server.once('close', function () {
      done();
    });
    server.close();
  });

  // test subject `http.Agent` instance
  var agent = new Agent(function (req, opts, fn) {
    if (!opts.port) opts.port = 80;
    var socket = net.connect(opts);
    fn(null, socket);
  });

  it('should work for basic HTTP requests', function (done) {
    // add HTTP server "request" listener
    var gotReq = false;
    server.once('request', function (req, res) {
      gotReq = true;
      res.setHeader('X-Foo', 'bar');
      res.setHeader('X-Url', req.url);
      res.end();
    });

    var info = url.parse('http://127.0.0.1:' + port + '/foo');
    info.agent = agent;
    http.get(info, function (res) {
      assert.equal('bar', res.headers['x-foo']);
      assert.equal('/foo', res.headers['x-url']);
      assert(gotReq);
      done();
    });
  });

  it('should set the `Connection: close` response header', function (done) {
    // add HTTP server "request" listener
    var gotReq = false;
    server.once('request', function (req, res) {
      gotReq = true;
      res.setHeader('X-Url', req.url);
      assert.equal('close', req.headers.connection);
      res.end();
    });

    var info = url.parse('http://127.0.0.1:' + port + '/bar');
    info.agent = agent;
    http.get(info, function (res) {
      assert.equal('/bar', res.headers['x-url']);
      assert.equal('close', res.headers.connection);
      assert(gotReq);
      done();
    });
  });
});

describe('"https" module', function () {
  var server;
  var port;

  // setup test HTTPS server
  before(function (done) {
    var options = {
      key: fs.readFileSync(__dirname + '/server.key'),
      cert: fs.readFileSync(__dirname + '/server.crt')
    };
    server = https.createServer(options);
    server.listen(0, function () {
      port = server.address().port;
      done();
    });
  });

  // shut down test HTTP server
  after(function (done) {
    server.once('close', function () {
      done();
    });
    server.close();
  });

  // test subject `http.Agent` instance
  var agent = new Agent(function (req, opts, fn) {
    if (!opts.port) opts.port = 443;
    opts.rejectUnauthorized = false;
    var socket = tls.connect(opts);
    fn(null, socket);
  });

  it('should work for basic HTTPS requests', function (done) {
    // add HTTPS server "request" listener
    var gotReq = false;
    server.once('request', function (req, res) {
      gotReq = true;
      res.setHeader('X-Foo', 'bar');
      res.setHeader('X-Url', req.url);
      res.end();
    });

    var info = url.parse('https://127.0.0.1:' + port + '/foo');
    info.agent = agent;
    info.rejectUnauthorized = false;
    https.get(info, function (res) {
      assert.equal('bar', res.headers['x-foo']);
      assert.equal('/foo', res.headers['x-url']);
      assert(gotReq);
      done();
    });
  });
});
