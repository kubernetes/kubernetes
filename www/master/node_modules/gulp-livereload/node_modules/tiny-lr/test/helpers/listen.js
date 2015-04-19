
var Server = require('../..').Server;
var request = require('supertest');

module.exports = function listen(opts) {
  opts = opts || {};

  return function _listen(done) {
    this.app = new Server();
    var srv = this.server = this.app.server;
    var ctx = this;
    this.server.listen(function (err) {
      if (err) return done(err);
      ctx.request = request(srv)
        .get('/')
        .expect(200, done);
    });
  };
};
