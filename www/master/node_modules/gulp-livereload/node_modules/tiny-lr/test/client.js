
var request = require('supertest');
var assert  = require('assert');
var parse   = require('url').parse;

var WebSocket = require('faye-websocket').Client;
var Server = require('..').Server;

var listen = require('./helpers/listen');

describe('tiny-lr', function() {

  before(listen());

  it('accepts ws clients', function(done) {
    var url = parse(this.request.url);
    var server = this.app;

    var ws = this.ws = new WebSocket('ws://' + url.host + '/livereload');

    ws.onopen = function(event) {
      var hello = {
        command: 'hello',
        protocols: ['http://livereload.com/protocols/official-7']
      };

      ws.send(JSON.stringify(hello));
    };

    ws.onmessage = function(event) {
      assert.deepEqual(event.data, JSON.stringify({
        command: 'hello',
        protocols: ['http://livereload.com/protocols/official-7'],
        serverName: 'tiny-lr'
      }));

      assert.ok(Object.keys(server.clients).length);
      done();
    };
  });

  it('properly cleans up established connection on exit', function(done) {
    var ws = this.ws;

    ws.onclose = done.bind(null, null);

    request(this.server)
      .get('/kill')
      .expect(200, function() {});
  });

});
