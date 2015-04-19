
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

(function (module, io, should) {

  module.exports = {

    'parse uri': function () {
      var http = io.util.parseUri('http://google.com')
        , https = io.util.parseUri('https://www.google.com:80')
        , query = io.util.parseUri('google.com:8080/foo/bar?foo=bar');

      http.protocol.should().eql('http');
      http.port.should().eql('');
      http.host.should().eql('google.com');
      https.protocol.should().eql('https');
      https.port.should().eql('80');
      https.host.should().eql('www.google.com');
      query.port.should().eql('8080');
      query.query.should().eql('foo=bar');
      query.path.should().eql('/foo/bar');
      query.relative.should().eql('/foo/bar?foo=bar');
    },

    'unique uri': function () {
      var protocol = io.util.parseUri('http://google.com')
        , noprotocol = io.util.parseUri('google.com')
        , https = io.util.parseUri('https://google.com')
        , path = io.util.parseUri('https://google.com/google.com/com/?foo=bar');

      if ('object' == typeof window) {
        io.util.uniqueUri(protocol).should().eql('http://google.com:3000');
        io.util.uniqueUri(noprotocol).should().eql('http://google.com:3000');
      } else {
        io.util.uniqueUri(protocol).should().eql('http://google.com:80');
        io.util.uniqueUri(noprotocol).should().eql('http://google.com:80');
      }

      io.util.uniqueUri(https).should().eql('https://google.com:443');
      io.util.uniqueUri(path).should().eql('https://google.com:443');
    },

    'chunk query string': function () {
      io.util.chunkQuery('foo=bar').should().be.a('object');
      io.util.chunkQuery('foo=bar').foo.should().eql('bar');
    },

    'merge query strings': function () {
      var base = io.util.query('foo=bar', 'foo=baz')
        , add = io.util.query('foo=bar', 'bar=foo')

      base.should().eql('?foo=baz');
      add.should().eql('?foo=bar&bar=foo');

      io.util.query('','').should().eql('');
      io.util.query('foo=bar', '').should().eql('?foo=bar');
      io.util.query('', 'foo=bar').should().eql('?foo=bar');
    },

    'request': function () {
      var type = typeof io.util.request();
      type.should().eql('object');
    },

    'is array': function () {
      io.util.isArray([]).should().be_true;
      io.util.isArray({}).should().be_false;
      io.util.isArray('str').should().be_false;
      io.util.isArray(new Date).should().be_false;
      io.util.isArray(true).should().be_false;
      io.util.isArray(arguments).should().be_false;
    },

    'merge, deep merge': function () {
      var start = {
            foo: 'bar'
          , bar: 'baz'
          }
        , duplicate = {
            foo: 'foo'
          , bar: 'bar'
          }
        , extra = {
            ping: 'pong'
          }
        , deep = {
            level1:{
              foo: 'bar'
            , level2: {
                foo: 'bar'
              ,  level3:{
                  foo: 'bar'
                , rescursive: deep
                }
              }
            }
          }
          // same structure, but changed names
        , deeper = {
            foo: 'bar'
          , level1:{
              foo: 'baz'
            , level2: {
                foo: 'foo'
              ,  level3:{
                  foo: 'pewpew'
                , rescursive: deep
                }
              }
            }
          };

      io.util.merge(start, duplicate);

      start.foo.should().eql('foo');
      start.bar.should().eql('bar');

      io.util.merge(start, extra);
      start.ping.should().eql('pong');
      start.foo.should().eql('foo');

      io.util.merge(deep, deeper);

      deep.foo.should().eql('bar');
      deep.level1.foo.should().eql('baz');
      deep.level1.level2.foo.should().eql('foo');
      deep.level1.level2.level3.foo.should().eql('pewpew');
    },

    'defer': function (next) {
      var now = +new Date;

      io.util.defer(function () {
        ((new Date - now) >= ( io.util.webkit ? 100 : 0 )).should().be_true();
        next();
      })
    },

    'indexOf': function () {
      var data = ['socket', 2, 3, 4, 'socket', 5, 6, 7, 'io'];
      io.util.indexOf(data, 'socket', 1).should().eql(4);
      io.util.indexOf(data, 'socket').should().eql(0);
      io.util.indexOf(data, 'waffles').should().eql(-1);
    }

  };

})(
    'undefined' == typeof module ? module = {} : module
  , 'undefined' == typeof io ? require('socket.io-client') : io
  , 'undefined' == typeof should ? require('should') : should
);
