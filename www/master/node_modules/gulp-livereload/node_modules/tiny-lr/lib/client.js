
var util      = require('util');
var events    = require('events');
var WebSocket = require('faye-websocket');

module.exports = Client;

function Client(req, socket, head, options) {
  options = this.options = options || {};
  this.ws = new WebSocket(req, socket, head);
  this.ws.onmessage = this.message.bind(this);
  this.ws.onclose = this.close.bind(this);
  this.id = this.uniqueId('ws');
}

util.inherits(Client, events.EventEmitter);

Client.prototype.message = function message(event) {
  var data = this.data(event);
  if(this[data.command]) return this[data.command](data);
};

Client.prototype.close = function close(event) {
  if(this.ws) {
    this.ws.close();
    this.ws = null;
  }

  this.emit('end', event);
};

// Commands

Client.prototype.hello = function hello() {
  this.send({
    command: 'hello',
    protocols: [
      'http://livereload.com/protocols/official-7'
    ],
    serverName: 'tiny-lr'
  });
};

Client.prototype.info = function info(data) {
  this.plugins = data.plugins;
  this.url = data.url;
};

// Server commands

Client.prototype.reload = function reload(files) {
  files.forEach(function(file) {
    this.send({
      command: 'reload',
      path: file,
      liveCSS: this.options.liveCSS !== false,
      liveJs: this.options.liveJs !== false,
      liveImg: this.options.liveImg !== false
    });
  }, this);
};

// Utilities

Client.prototype.data = function _data(event) {
  var data = {};
  try {
    data = JSON.parse(event.data);
  } catch (e) {}
  return data;
};

Client.prototype.send = function send(data) {
  this.ws.send(JSON.stringify(data));
};

var idCounter = 0;
Client.prototype.uniqueId = function uniqueId(prefix) {
  var id = idCounter++;
  return prefix ? prefix + id : id;
};
