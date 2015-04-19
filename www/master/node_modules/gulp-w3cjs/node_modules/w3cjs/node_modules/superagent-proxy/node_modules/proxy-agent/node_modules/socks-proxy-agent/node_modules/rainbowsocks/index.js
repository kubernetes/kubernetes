
/**
 * Module Dependencies
 */

var net = require('net'),
    buffer = require('buffer'),
    debug = require('debug')('RainbowSocks'),
    EventEmitter = require('events').EventEmitter,
    util = require('util');

/**
 * Socks 4a Client "Class"
 * @param  {Number} port Socks4A Server Port
 * @param  {String} host Socks4A Server Host
 * @return {RainbowSocks}
 */
var RainbowSocks = function(port, host) {
  var _this = this;
  this.socket = net.connect({
    host: host || '127.0.0.1',
    port: port
  });
  this.socket.on('connect', function() {
    _this.emit('connect');
  });
  this.socket.on('error', function(err) {
    _this.emit('error', err);
  });
  return this;
};

util.inherits(RainbowSocks, EventEmitter);

/**
 * SOCKS4A Connect Request
 * @param  {String}   host     IP/Domain of desired destination
 * @param  {Number}   port     Port of desired destination
 * @param  {Function} callback (err, socket)
 */
RainbowSocks.prototype.connect = function(host, port, callback) {
  debug('RainbowSocks.connect(%j, %j)', host, port);
  this.request(new Buffer('01', 'hex'), host, port, callback);
};

/**
 * SOCKS4A Bind Request
 * @param  {String}   host     IP/Domain of desired destination
 * @param  {Number}   port     Port of desired destination
 * @param  {Function} callback (err, socket)
 */
RainbowSocks.prototype.bind = function(host, port, callback) {
  debug('RainbowSocks.connect(%j, %j)', host, port);
  this.request(new Buffer('02', 'hex'), host, port, callback);
};

/**
 * SOCKS4A Raw Request Generator
 * @param  {Buffer}   cmdBuf   repesenting a valid SOCKS4A command code
 * @param  {String}   domain   IP/Domain of desired destination
 * @param  {Number}   port     Port of desired destination
 * @param  {Function} callback (err, socker)
 */
RainbowSocks.prototype.request = function(cmdBuf, domain, port, callback) {
  debug('RainbowSocks.request(%j, %j, %j)', cmdBuf.toString('hex'), domain, port);
  var _this = this;

  var portBuf = new Buffer(2);
  portBuf.writeUInt16BE(port, 0);

  var domainBuf = new Buffer(domain, 'utf8');

  this.socket.write(Buffer.concat([
    new Buffer('04', 'hex'), // SOCKS VERSION
    cmdBuf,
    portBuf,
    new Buffer('00000001', 'hex'), // Invalid identify as SOCKS4A rather than 4
    new Buffer('FFFFFF00', 'hex'), // User ID
    domainBuf, new Buffer('00', 'hex') // domain name of the host we want to contact, variable length, terminated with a null
  ]));

  this.socket.once('data', function(data) {
    if(data[0] !== 0) return callback(new Error('response missing null byte'));
    if(data[1] == 0x5a) return callback(null, _this.socket);
    else if(data[1] == 0x5b) return callback(new Error('request rejected or failed'));
    else if(data[1] == 0x5c) return callback(new Error('request failed because client is not running identd (or not reachable from the server)'));
    else if(data[1] == 0x5d) return callback(new Error('request failed because client\'s identd could not confirm the user ID string in the request'));
    else return callback(new Error('unknown response status'));
  });
};

module.exports = RainbowSocks;
