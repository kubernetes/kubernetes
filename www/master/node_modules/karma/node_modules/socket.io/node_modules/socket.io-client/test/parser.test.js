
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

(function (module, io, should) {

  var parser = io.parser;

  module.exports = {

    'decoding error packet': function () {
      parser.decodePacket('7:::').should().eql({
          type: 'error'
        , reason: ''
        , advice: ''
        , endpoint: ''
      });
    },

    'decoding error packet with reason': function () {
      parser.decodePacket('7:::0').should().eql({
          type: 'error'
        , reason: 'transport not supported'
        , advice: ''
        , endpoint: ''
      });
    },

    'decoding error packet with reason and advice': function () {
      parser.decodePacket('7:::2+0').should().eql({
          type: 'error'
        , reason: 'unauthorized'
        , advice: 'reconnect'
        , endpoint: ''
      });
    },

    'decoding error packet with endpoint': function () {
      parser.decodePacket('7::/woot').should().eql({
          type: 'error'
        , reason: ''
        , advice: ''
        , endpoint: '/woot'
      });
    },

    'decoding ack packet': function () {
      parser.decodePacket('6:::140').should().eql({
          type: 'ack'
        , ackId: '140'
        , endpoint: ''
        , args: []
      });
    },

    'decoding ack packet with args': function () {
      parser.decodePacket('6:::12+["woot","wa"]').should().eql({
          type: 'ack'
        , ackId: '12'
        , endpoint: ''
        , args: ['woot', 'wa']
      });
    },

    'decoding ack packet with bad json': function () {
      var thrown = false;

      try {
        parser.decodePacket('6:::1+{"++]').should().eql({
            type: 'ack'
          , ackId: '1'
          , endpoint: ''
          , args: []
        });
      } catch (e) {
        thrown = true;
      }

      thrown.should().be_false;
    },

    'decoding json packet': function () {
      parser.decodePacket('4:::"2"').should().eql({
          type: 'json'
        , endpoint: ''
        , data: '2'
      });
    },

    'decoding json packet with message id and ack data': function () {
      parser.decodePacket('4:1+::{"a":"b"}').should().eql({
          type: 'json'
        , id: 1
        , ack: 'data'
        , endpoint: ''
        , data: { a: 'b' }
      });
    },

    'decoding an event packet': function () {
      parser.decodePacket('5:::{"name":"woot"}').should().eql({
          type: 'event'
        , name: 'woot'
        , endpoint: ''
        , args: []
      });
    },

    'decoding an event packet with message id and ack': function () {
      parser.decodePacket('5:1+::{"name":"tobi"}').should().eql({
          type: 'event'
        , id: 1
        , ack: 'data'
        , endpoint: ''
        , name: 'tobi'
        , args: []
      });
    },

    'decoding an event packet with data': function () {
      parser.decodePacket('5:::{"name":"edwald","args":[{"a": "b"},2,"3"]}')
      .should().eql({
          type: 'event'
        , name: 'edwald'
        , endpoint: ''
        , args: [{a: 'b'}, 2, '3']
      });
    },

    'decoding a message packet': function () {
      parser.decodePacket('3:::woot').should().eql({
          type: 'message'
        , endpoint: ''
        , data: 'woot'
      });
    },

    'decoding a message packet with id and endpoint': function () {
      parser.decodePacket('3:5:/tobi').should().eql({
          type: 'message'
        , id: 5
        , ack: true
        , endpoint: '/tobi'
        , data: ''
      });
    },

    'decoding a heartbeat packet': function () {
      parser.decodePacket('2:::').should().eql({
          type: 'heartbeat'
        , endpoint: ''
      });
    },

    'decoding a connection packet': function () {
      parser.decodePacket('1::/tobi').should().eql({
          type: 'connect'
        , endpoint: '/tobi'
        , qs: ''
      });
    },

    'decoding a connection packet with query string': function () {
      parser.decodePacket('1::/test:?test=1').should().eql({
          type: 'connect'
        , endpoint: '/test'
        , qs: '?test=1'
      });
    },

    'decoding a disconnection packet': function () {
      parser.decodePacket('0::/woot').should().eql({
          type: 'disconnect'
        , endpoint: '/woot'
      });
    },

    'encoding error packet': function () {
      parser.encodePacket({
          type: 'error'
        , reason: ''
        , advice: ''
        , endpoint: ''
      }).should().eql('7::');
    },

    'encoding error packet with reason': function () {
      parser.encodePacket({
          type: 'error'
        , reason: 'transport not supported'
        , advice: ''
        , endpoint: ''
      }).should().eql('7:::0');
    },

    'encoding error packet with reason and advice': function () {
      parser.encodePacket({
          type: 'error'
        , reason: 'unauthorized'
        , advice: 'reconnect'
        , endpoint: ''
      }).should().eql('7:::2+0');
    },

    'encoding error packet with endpoint': function () {
      parser.encodePacket({
          type: 'error'
        , reason: ''
        , advice: ''
        , endpoint: '/woot'
      }).should().eql('7::/woot');
    },

    'encoding ack packet': function () {
      parser.encodePacket({
          type: 'ack'
        , ackId: '140'
        , endpoint: ''
        , args: []
      }).should().eql('6:::140');
    },

    'encoding ack packet with args': function () {
      parser.encodePacket({
          type: 'ack'
        , ackId: '12'
        , endpoint: ''
        , args: ['woot', 'wa']
      }).should().eql('6:::12+["woot","wa"]');
    },

    'encoding json packet': function () {
      parser.encodePacket({
          type: 'json'
        , endpoint: ''
        , data: '2'
      }).should().eql('4:::"2"');
    },

    'encoding json packet with message id and ack data': function () {
      parser.encodePacket({
          type: 'json'
        , id: 1
        , ack: 'data'
        , endpoint: ''
        , data: { a: 'b' }
      }).should().eql('4:1+::{"a":"b"}');
    },

    'encoding an event packet': function () {
      parser.encodePacket({
          type: 'event'
        , name: 'woot'
        , endpoint: ''
        , args: []
      }).should().eql('5:::{"name":"woot"}');
    },

    'encoding an event packet with message id and ack': function () {
      parser.encodePacket({
          type: 'event'
        , id: 1
        , ack: 'data'
        , endpoint: ''
        , name: 'tobi'
        , args: []
      }).should().eql('5:1+::{"name":"tobi"}');
    },

    'encoding an event packet with data': function () {
      parser.encodePacket({
          type: 'event'
        , name: 'edwald'
        , endpoint: ''
        , args: [{a: 'b'}, 2, '3']
      }).should().eql('5:::{"name":"edwald","args":[{"a":"b"},2,"3"]}');
    },

    'encoding a message packet': function () {
      parser.encodePacket({
          type: 'message'
        , endpoint: ''
        , data: 'woot'
      }).should().eql('3:::woot');
    },

    'encoding a message packet with id and endpoint': function () {
      parser.encodePacket({
          type: 'message'
        , id: 5
        , ack: true
        , endpoint: '/tobi'
        , data: ''
      }).should().eql('3:5:/tobi');
    },

    'encoding a heartbeat packet': function () {
      parser.encodePacket({
          type: 'heartbeat'
        , endpoint: ''
      }).should().eql('2::');
    },

    'encoding a connection packet': function () {
      parser.encodePacket({
          type: 'connect'
        , endpoint: '/tobi'
        , qs: ''
      }).should().eql('1::/tobi');
    },

    'encoding a connection packet with query string': function () {
      parser.encodePacket({
          type: 'connect'
        , endpoint: '/test'
        , qs: '?test=1'
      }).should().eql('1::/test:?test=1');
    },

    'encoding a disconnection packet': function () {
      parser.encodePacket({
          type: 'disconnect'
        , endpoint: '/woot'
      }).should().eql('0::/woot');
    },

    'test decoding a payload': function () {
      parser.decodePayload('\ufffd5\ufffd3:::5\ufffd7\ufffd3:::53d'
        + '\ufffd3\ufffd0::').should().eql([
          { type: 'message', data: '5', endpoint: '' }
        , { type: 'message', data: '53d', endpoint: '' }
        , { type: 'disconnect', endpoint: '' }
      ]);
    },

    'test encoding a payload': function () {
      parser.encodePayload([
          parser.encodePacket({ type: 'message', data: '5', endpoint: '' })
        , parser.encodePacket({ type: 'message', data: '53d', endpoint: '' })
      ]).should().eql('\ufffd5\ufffd3:::5\ufffd7\ufffd3:::53d')
    },

    'test decoding newline': function () {
      parser.decodePacket('3:::\n').should().eql({
          type: 'message'
        , endpoint: ''
        , data: '\n'
      });
    }

  };

})(
    'undefined' == typeof module ? module = {} : module
  , 'undefined' == typeof io ? require('socket.io-client') : io
  , 'undefined' == typeof should ? require('should') : should
);
