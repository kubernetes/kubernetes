/*
 * routing-tabletest.js: Test for proxying `socket.io` and raw `WebSocket` requests using a ProxyTable.
 *
 * (C) 2010 Nodejitsu Inc.
 * MIT LICENCE
 *
 */

var vows = require('vows'),
    macros = require('../macros'),
    helpers = require('../helpers/index');

vows.describe(helpers.describe('routing-proxy', 'ws')).addBatch({
  "With a valid target server": {
    "and no latency": {
      "using ws": macros.ws.assertProxied(),
      "using socket.io": macros.ws.assertProxied({
        raw: true
      }),
    },
    // "and latency": macros.websocket.assertProxied({
    //   latency: 2000
    // })
  }
}).export(module);