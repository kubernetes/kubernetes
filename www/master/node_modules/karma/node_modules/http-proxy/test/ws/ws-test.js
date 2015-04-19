/*
 * ws-test.js: Tests for proxying raw Websocket requests.
 *
 * (C) 2010 Nodejitsu Inc.
 * MIT LICENCE
 *
 */

var vows = require('vows'),
    macros = require('../macros'),
    helpers = require('../helpers/index');

vows.describe(helpers.describe('websocket', 'ws')).addBatch({
  "With a valid target server": {
    "and no latency": macros.ws.assertProxied({
      raw: true
    }),
    // "and latency": macros.ws.assertProxied({
    //   raw: true,
    //   latency: 2000
    // })
  }
}).export(module);