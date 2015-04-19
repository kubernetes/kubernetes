
/**
 * Module dependencies.
 */

var fs = require('fs');
var url = require('url');
var http = require('http');
var https = require('https');
var assert = require('assert');
var Proxy = require('proxy');
var Semver = require('semver');
var version = new Semver(process.version);
var HttpsProxyAgent = require('../');

describe('HttpsProxyAgent', function () {

  var server;
  var serverPort;

  var sslServer;
  var sslServerPort;

  var proxy;
  var proxyPort;

  var sslProxy;
  var sslProxyPort;

  before(function (done) {
    // setup target HTTP server
    server = http.createServer();
    server.listen(function () {
      serverPort = server.address().port;
      done();
    });
  });

  before(function (done) {
    // setup HTTP proxy server
    proxy = Proxy();
    proxy.listen(function () {
      proxyPort = proxy.address().port;
      done();
    });
  });

  before(function (done) {
    // setup target HTTPS server
    var options = {
      key: fs.readFileSync(__dirname + '/server.key'),
      cert: fs.readFileSync(__dirname + '/server.crt')
    };
    sslServer = https.createServer(options);
    sslServer.listen(function () {
      sslServerPort = sslServer.address().port;
      done();
    });
  });

  before(function (done) {
    // setup SSL HTTP proxy server
    var options = {
      key: fs.readFileSync(__dirname + '/server.key'),
      cert: fs.readFileSync(__dirname + '/server.crt')
    };
    sslProxy = Proxy(https.createServer(options));
    sslProxy.listen(function () {
      sslProxyPort = sslProxy.address().port;
      done();
    });
  });

  // shut down test HTTP server
  after(function (done) {
    server.once('close', function () { done(); });
    server.close();
  });

  after(function (done) {
    proxy.once('close', function () { done(); });
    proxy.close();
  });

  after(function (done) {
    sslServer.once('close', function () { done(); });
    sslServer.close();
  });

  after(function (done) {
    sslProxy.once('close', function () { done(); });
    sslProxy.close();
  });

  describe('constructor', function () {
    it('should throw an Error if no "proxy" argument is given', function () {
      assert.throws(function () {
        new HttpsProxyAgent();
      });
    });
    it('should accept a "string" proxy argument', function () {
      var agent = new HttpsProxyAgent('http://127.0.0.1:' + proxyPort);
      assert.equal('127.0.0.1', agent.proxy.host);
      assert.equal(proxyPort, agent.proxy.port);
    });
    it('should accept a `url.parse()` result object argument', function () {
      var opts = url.parse('http://127.0.0.1:' + proxyPort);
      var agent = new HttpsProxyAgent(opts);
      assert.equal('127.0.0.1', agent.proxy.host);
      assert.equal(proxyPort, agent.proxy.port);
    });
    describe('secureEndpoint', function () {
      it('should default to `true`', function () {
        var agent = new HttpsProxyAgent('http://127.0.0.1:' + proxyPort);
        assert.equal(true, agent.secureEndpoint);
      });
      it('should be `false` when passed in as an option', function () {
        var opts = url.parse('http://127.0.0.1:' + proxyPort);
        opts.secureEndpoint = false;
        var agent = new HttpsProxyAgent(opts);
        assert.equal(false, agent.secureEndpoint);
      });
      it('should be `true` when passed in as an option', function () {
        var opts = url.parse('http://127.0.0.1:' + proxyPort);
        opts.secureEndpoint = true;
        var agent = new HttpsProxyAgent(opts);
        assert.equal(true, agent.secureEndpoint);
      });
    });
    describe('secureProxy', function () {
      it('should default to `false`', function () {
        var agent = new HttpsProxyAgent({ port: proxyPort });
        assert.equal(false, agent.secureProxy);
      });
      it('should be `false` when "http:" protocol is used', function () {
        var agent = new HttpsProxyAgent({ port: proxyPort, protocol: 'http:' });
        assert.equal(false, agent.secureProxy);
      });
      it('should be `true` when "https:" protocol is used', function () {
        var agent = new HttpsProxyAgent({ port: proxyPort, protocol: 'https:' });
        assert.equal(true, agent.secureProxy);
      });
      it('should be `true` when "https" protocol is used', function () {
        var agent = new HttpsProxyAgent({ port: proxyPort, protocol: 'https' });
        assert.equal(true, agent.secureProxy);
      });
    });
  });

  describe('"http" module', function () {

    beforeEach(function () {
      delete proxy.authenticate;
    });

    it('should receive the 407 authorization code on the `http.ClientResponse`', function (done) {
      // set a proxy authentication function for this test
      proxy.authenticate = function (req, fn) {
        // reject all requests
        fn(null, false);
      };

      var proxyUri = process.env.HTTP_PROXY || process.env.http_proxy || 'http://127.0.0.1:' + proxyPort;
      var agent = new HttpsProxyAgent(proxyUri);

      var opts = {};
      // `host` and `port` don't really matter since the proxy will reject anyways
      opts.host = '127.0.0.1';
      opts.port = 80;
      opts.agent = agent;

      var req = http.get(opts, function (res) {
        assert.equal(407, res.statusCode);
        assert('proxy-authenticate' in res.headers);
        done();
      });
    });
    it('should emit an "error" event on the `http.ClientRequest` if the proxy does not exist', function (done) {
      // port 4 is a reserved, but "unassigned" port
      var proxyUri = 'http://127.0.0.1:4';
      var agent = new HttpsProxyAgent(proxyUri);

      var opts = url.parse('http://nodejs.org');
      opts.agent = agent;

      var req = http.get(opts);
      req.once('error', function (err) {
        assert.equal('ECONNREFUSED', err.code);
        req.abort();
        done();
      });
    });
  });

  describe('"https" module', function () {
    it('should work over an HTTP proxy', function (done) {
      // set HTTP "request" event handler for this test
      sslServer.once('request', function (req, res) {
        res.end(JSON.stringify(req.headers));
      });

      var proxy = process.env.HTTP_PROXY || process.env.http_proxy || 'http://127.0.0.1:' + proxyPort;
      proxy = url.parse(proxy);
      // `rejectUnauthorized` shoudn't *technically* be necessary here,
      // but up until node v0.11.6, the `http.Agent` class didn't have
      // access to the *entire* request "options" object. Thus,
      // `https-proxy-agent` will *also* merge in options you pass here
      // to the destination endpoints…
      proxy.rejectUnauthorized = false;
      var agent = new HttpsProxyAgent(proxy);

      var opts = url.parse('https://127.0.0.1:' + sslServerPort);
      opts.rejectUnauthorized = false;
      opts.agent = agent;

      https.get(opts, function (res) {
        var data = '';
        res.setEncoding('utf8');
        res.on('data', function (b) {
          data += b;
        });
        res.on('end', function () {
          data = JSON.parse(data);
          assert.equal('127.0.0.1:' + sslServerPort, data.host);
          done();
        });
      });
    });

    if (version.compare('0.11.3') < 0 || version.compare('0.11.8') >= 0) {
      // This test is disabled on node >= 0.11.3 && < 0.11.8, since it segfaults :(
      // See: https://github.com/joyent/node/issues/6204

      it('should work over an HTTPS proxy', function (done) {
        // set HTTP "request" event handler for this test
        sslServer.once('request', function (req, res) {
          res.end(JSON.stringify(req.headers));
        });

        var proxy = process.env.HTTPS_PROXY || process.env.https_proxy || 'https://127.0.0.1:' + sslProxyPort;
        proxy = url.parse(proxy);
        // `rejectUnauthorized` is actually necessary this time since the HTTPS
        // proxy server itself is using a self-signed SSL certificate…
        proxy.rejectUnauthorized = false;
        var agent = new HttpsProxyAgent(proxy);

        var opts = url.parse('https://127.0.0.1:' + sslServerPort);
        opts.agent = agent;
        opts.rejectUnauthorized = false;

        https.get(opts, function (res) {
          var data = '';
          res.setEncoding('utf8');
          res.on('data', function (b) {
            data += b;
          });
          res.on('end', function () {
            data = JSON.parse(data);
            assert.equal('127.0.0.1:' + sslServerPort, data.host);
            done();
          });
        });
      });
    } else {
      it('should work over an HTTPS proxy');
    }

  });

});
