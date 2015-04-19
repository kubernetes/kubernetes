/*
 * routing-table-test.js: Tests for the proxying using the ProxyTable object.
 *
 * (C) 2010, Charlie Robbins
 *
 */

var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    async = require('async'),
    request = require('request'),
    vows = require('vows'),
    macros = require('../macros'),
    helpers = require('../helpers');

var routeFile = path.join(__dirname, 'config.json');

vows.describe(helpers.describe('routing-table')).addBatch({
  "With a routing table": {
    "with latency": macros.http.assertProxiedToRoutes({
      latency: 2000,
      routes: {
        "icanhaz.com": "127.0.0.1:{PORT}",
        "latency.com": "127.0.0.1:{PORT}"
      }
    }),
    "addHost() / removeHost()": macros.http.assertDynamicProxy({
      hostnameOnly: true,
      routes: {
        "static.com":  "127.0.0.1:{PORT}",
        "removed.com": "127.0.0.1:{PORT}"
      }
    }, {
      add: [{ host: 'dynamic1.com', target: '127.0.0.1:' }],
      drop: ['removed.com']
    }),
    "using RegExp": macros.http.assertProxiedToRoutes({
      routes: {
        "foo.com": "127.0.0.1:{PORT}",
        "bar.com": "127.0.0.1:{PORT}",
        "baz.com/taco": "127.0.0.1:{PORT}",
        "pizza.com/taco/muffins": "127.0.0.1:{PORT}",
        "blah.com/me": "127.0.0.1:{PORT}/remapped",
        "bleh.com/remap/this": "127.0.0.1:{PORT}/remap/remapped",
        "test.com/double/tap": "127.0.0.1:{PORT}/remap"
      }
    }),
    "using hostnameOnly": macros.http.assertProxiedToRoutes({
      hostnameOnly: true,
      routes: {
        "foo.com": "127.0.0.1:{PORT}",
        "bar.com": "127.0.0.1:{PORT}"
      }
    }),
    "using pathnameOnly": macros.http.assertProxiedToRoutes({
      pathnameOnly: true,
      routes: {
        "/foo": "127.0.0.1:{PORT}",
        "/bar": "127.0.0.1:{PORT}",
        "/pizza": "127.0.0.1:{PORT}"
      }
    }),
    "using a routing file": macros.http.assertProxiedToRoutes({
      filename: routeFile,
      routes: {
        "foo.com": "127.0.0.1:{PORT}",
        "bar.com": "127.0.0.1:{PORT}"
      }
    }, {
      "after the file has been modified": {
        topic: function () {
          var config = JSON.parse(fs.readFileSync(routeFile, 'utf8')),
              protocol = helpers.protocols.proxy,
              port = helpers.nextPort,
              that = this;

          config.router['dynamic.com'] = "127.0.0.1:" + port;
          fs.writeFileSync(routeFile, JSON.stringify(config));

          async.parallel([
            function waitForRoutes(next) {
              that.proxyServer.on('routes', next);
            },
            async.apply(
              helpers.http.createServer,
              {
                port: port,
                output: 'hello from dynamic.com'
              }
            )
          ], function () {
            request({
              uri: protocol + '://127.0.0.1:' + that.port,
              headers: {
                host: 'dynamic.com'
              }
            }, that.callback);
          });
        },
        "should receive 'hello from dynamic.com'": function (err, res, body) {
          assert.equal(body, 'hello from dynamic.com');
        }
      }
    })
  }
}).export(module);
