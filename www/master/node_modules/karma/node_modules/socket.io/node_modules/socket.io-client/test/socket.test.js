
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

(function (module, io, should) {

  if ('object' == typeof global) {
    return module.exports = { '': function () {} };
  }

  module.exports = {

    'test connecting the socket and disconnecting': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('connect', function () {
        socket.disconnect();
        next();
      });
    },

    'test receiving messages': function (next) {
      var socket = create()
        , connected = false
        , messages = 0;

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('connect', function () {
        connected = true;
      });

      socket.on('message', function (i) {
        String(++messages).should().equal(i);
      });

      socket.on('disconnect', function (reason) {
        connected.should().be_true;
        messages.should().equal(3);
        reason.should().eql('booted');
        next();
      });
    },

    'test sending messages': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('connect', function () {
        socket.send('echo');

        socket.on('message', function (msg) {
          msg.should().equal('echo');
          socket.disconnect();
          next();
        });
      });
    },

    'test manual buffer flushing': function (next) {
      var socket = create();

      socket.socket.options['manualFlush'] = true;

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('connect', function () {
        socket.socket.connected = false;
        socket.send('buffered');
        socket.socket.onConnect();
        socket.socket.flushBuffer();

        socket.on('message', function (msg) {
          msg.should().equal('buffered');
          socket.disconnect();
          next();
        });
      });
    },

    'test automatic buffer flushing': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('connect', function () {
        socket.socket.connected = false;
        socket.send('buffered');
        socket.socket.onConnect();

        socket.on('message', function (msg) {
          msg.should().equal('buffered');
          socket.disconnect();
          next();
        });
      });
    },

    'test acks sent from client': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('connect', function () {
        socket.on('message', function (msg) {
          if ('tobi 2' == msg) {
            socket.disconnect();
            next();
          }
        });
      });
    },

    'test acks sent from server': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('connect', function () {
        socket.send('ooo', function () {
          socket.disconnect();
          next();
        });
      });
    },

    'test connecting to namespaces': function (next) {
      var io = create()
        , socket = io.socket
        , namespaces = 2
        , connect = 0;

      function finish () {
        socket.of('').disconnect();
        connect.should().equal(3);
        next();
      }

      socket.on('connect', function(){
        connect++;
      });

      socket.of('/woot').on('connect', function () {
        connect++;
      }).on('message', function (msg) {
        msg.should().equal('connected to woot');
        --namespaces || finish();
      }).on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.of('/chat').on('connect', function () {
        connect++;
      }).on('message', function (msg) {
        msg.should().equal('connected to chat');
        --namespaces || finish();
      }).on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });
    },

    'test disconnecting from namespaces': function (next) {
      var socket = create().socket
        , namespaces = 2
        , disconnections = 0;

      function finish () {
        socket.of('').disconnect();
        next();
      };

      socket.of('/a').on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.of('/a').on('connect', function () {
        socket.of('/a').disconnect();
      });

      socket.of('/a').on('disconnect', function () {
        --namespaces || finish();
      });

      socket.of('/b').on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.of('/b').on('connect', function () {
        socket.of('/b').disconnect();
      });

      socket.of('/b').on('disconnect', function () {
        --namespaces || finish();
      });
    },

    'test authorizing for namespaces': function (next) {
      var socket = create().socket

      function finish () {
        socket.of('').disconnect();
        next();
      };

      socket.of('/a')
        .on('connect_failed', function (msg) {
          next();
        })
        .on('error', function (msg) {
          throw new Error(msg || 'Received an error');
        });
    },

    'test sending json from server': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('message', function (msg) {
        msg.should().eql(3141592);
        socket.disconnect();
        next();
      });
    },

    'test sending json from client': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.json.send([1, 2, 3]);
      socket.on('message', function (msg) {
        msg.should().equal('echo');
        socket.disconnect();
        next();
      });
    },

    'test emitting an event from server': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('woot', function () {
        socket.disconnect();
        next();
      });
    },

    'test emitting an event to server': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.emit('woot');
      socket.on('echo', function () {
        socket.disconnect();
        next();
      })
    },

    'test emitting multiple events at once to the server': function (next) {
      var socket = create();

      socket.on('connect', function () {
        socket.emit('print', 'foo');
        socket.emit('print', 'bar');
      });

      socket.on('done', function () {
        socket.disconnect();
        next();
      });
    },

    'test emitting an event from server and sending back data': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('woot', function (a, fn) {
        a.should().eql(1);
        fn('test');

        socket.on('done', function () {
          socket.disconnect();
          next();
        });
      });
    },

    'test emitting an event to server and sending back data': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.emit('tobi', 1, 2, function (a) {
        a.should().eql({ hello: 'world' });
        socket.disconnect();
        next();
      });
    },

    'test encoding a payload': function (next) {
      var socket = create('/woot');

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('connect', function () {
        socket.socket.setBuffer(true);
        socket.send('ñ');
        socket.send('ñ');
        socket.send('ñ');
        socket.send('ñ');
        socket.socket.setBuffer(false);
      });

      socket.on('done', function () {
        socket.disconnect();
        next();
      });
    },

    'test sending query strings to the server': function (next) {
      var socket = create('?foo=bar');

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.on('message', function (data) {
        data.query.foo.should().eql('bar');

        socket.disconnect();
        next();
      });
    },

    'test sending newline': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.send('\n');

      socket.on('done', function () {
        socket.disconnect();
        next();
      });
    },

    'test sending unicode': function (next) {
      var socket = create();

      socket.on('error', function (msg) {
        throw new Error(msg || 'Received an error');
      });

      socket.json.send({ test: "☃" });

      socket.on('done', function () {
        socket.disconnect();
        next();
      });
    },

    'test webworker connection': function (next) {
      if (!window.Worker) {
        return next();
      }

      var worker = new Worker('/test/worker.js');
      worker.postMessage(uri());
      worker.onmessage = function (ev) {
        if ('done!' == ev.data) return next();
        throw new Error('Unexpected message: ' + ev.data);
      }
    }

  };

})(
    'undefined' == typeof module ? module = {} : module
  , 'undefined' == typeof io ? require('socket.io-client') : io
  , 'undefined' == typeof should ? require('should-browser') : should
);
