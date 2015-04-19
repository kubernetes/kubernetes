/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Test dependencies.
 */

var builder = require('../../bin/builder')
  , common = require('./builder.common')
  , should = require('should');

/**
 * Tests.
 */

module.exports = {

  'version number': function () {
    builder.version.should().match(/([0-9]+)\.([0-9]+)\.([0-9]+)/);
    builder.version.should().equal(require('../../lib/io').version);
  },

  'production build LOC': function () {
    builder(function (err, result) {
      should.strictEqual(err, null)

      var lines = result.split('\n');
      lines.length.should().be.below(5);
      lines[0].should().match(/production/gi);
      Buffer.byteLength(result).should().be.below(43000);
    });
  },

  'development build LOC': function () {
    builder({ minify: false }, function (err, result) {
      should.strictEqual(err, null)

      var lines = result.split('\n');
      lines.length.should().be.above(5);
      lines[0].should().match(/development/gi);
      Buffer.byteLength(result).should().be.above(35000);
    });
  },

  'default builds': function () {
    builder(function (err, result) {
      should.strictEqual(err, null);

      var io = common.execute(result).io
        , transports = Object.keys(io.Transport)
        , defaults = Object.keys(builder.transports);

      /* XHR transport is private, but still available */
      transports.length.should().be.equal(defaults.length + 1);

      defaults.forEach(function (transport) {
        transports.indexOf(transport).should().be.above(-1);
      })
    });
  },

  'custom build': function () {
    builder(['websocket'], function (err, result) {
      should.strictEqual(err, null);

      var io = common.execute(result).io
        , transports = Object.keys(io.Transport);

      transports.should().have.length(1);
      transports[0].should().eql('websocket');
    });
  },

  'custom code': function () {
    var custom = 'var hello = "world";';
    builder({ custom: [custom], minify: false }, function (err, result) {
      should.strictEqual(err, null);

      result.should().include.string(custom);
    });
  },

  'node if': function () {
    var custom = '// if node \nvar hello = "world";\n'
      + '// end node\nvar pew = "pew";';

    builder({ custom: [custom], minify: false }, function (err, result) {
      should.strictEqual(err, null);

      result.should().not.include.string(custom);
      result.should().not.include.string('// if node');
      result.should().not.include.string('// end node');
      result.should().not.include.string('"world"');

      result.should().include.string('var pew = "pew"');
    });
  },

  'preserve the encoding during minification': function () {
    builder(function (err, result) {
      should.strictEqual(err, null);

      result.should().match(/(\\ufffd)/g);
    })
  },

  'globals': function () {
    builder(function (err, result) {
      should.strictEqual(err, null);

      var io = common.execute(result)
        , env = common.env()
        , allowed = ['io', 'swfobject', 'WEB_SOCKET_DISABLE_AUTO_INITIALIZATION'];

      Array.prototype.push.apply(allowed, Object.keys(env));

      Object.keys(io).forEach(function (global) {
        var index = allowed.indexOf(global);

        // the global is not allowed!
        if (!~index) {
          throw new Error('Global leak: ' + global);
        }
      });
    })
  }

};
