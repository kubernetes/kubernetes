
/**
 * Module dependencies.
 */

var net = require('net');
var url = require('url');
var http = require('http');
var https = require('https');
var assert = require('assert');
var SocksProxyAgent = require('../');

describe('SocksProxyAgent', function () {

  // adjusting the "slow" and "timeout" values because I run the
  // tests against the Tor SOCKS proxy
  this.slow(5000);
  this.timeout(10000);

  var proxy = process.env.SOCKS_PROXY || process.env.socks_proxy || 'socks://127.0.0.1:9050';

  it('should work against an HTTP endpoint', function (done) {
    var agent = new SocksProxyAgent(proxy);
    var link = 'http://jsonip.com/';
    var opts = url.parse(link);
    opts.agent = agent;
    http.get(opts, function (res) {
      var data = '';
      res.setEncoding('utf8');
      res.on('data', function (b) {
        data += b;
      });
      res.on('end', function () {
        data = JSON.parse(data);
        assert('ip' in data);
        assert(net.isIP(data.ip));
        done();
      });
    });
  });

  it('should work against an HTTPS endpoint', function (done) {
    var agent = new SocksProxyAgent(proxy, true);
    var link = 'https://graph.facebook.com/tootallnate';
    var opts = url.parse(link);
    opts.agent = agent;
    https.get(opts, function (res) {
      var data = '';
      res.setEncoding('utf8');
      res.on('data', function (b) {
        data += b;
      });
      res.on('end', function () {
        data = JSON.parse(data);
        assert.equal('tootallnate', data.username);
        done();
      });
    });
  });

});
