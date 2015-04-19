
var assert       = require('assert');
var RainbowSocks = require('./');

describe('RainbowSocks', function() {
  var sock = new RainbowSocks(9050);

  describe('creating a client', function() {
    it('should emit `connect`', function(done) {
      sock = new RainbowSocks(9050);
      sock.on('connect', done);
    });
    it('should error when no server', function(done) {
      var noSock = new RainbowSocks(9051);
      noSock.on('error', function(err) {
        assert(err instanceof Error);
        done();
      });
    });
  });

  describe('#connect', function() {
    it('should open a socket', function(done) {
      sock.connect('www.google.com', 80, function(err, socket) {
        if(err) return next(err);
        assert(socket.readable);
        assert(socket.writable);
        done();
      });
    });
  });

});
