
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

(function (module, io, should) {

  module.exports = {
    
    'client version number': function () {
      io.version.should().match(/([0-9]+)\.([0-9]+)\.([0-9]+)/);
    },

    'socket.io protocol version': function () {
      io.protocol.should().be.a('number');
      io.protocol.toString().should().match(/^\d+$/);
    },

    'socket.io available transports': function () {
      (io.transports.length > 0).should().be_true;
    }

  };

})(
    'undefined' == typeof module ? module = {} : module
  , 'undefined' == typeof io ? require('socket.io-client') : io
  , 'undefined' == typeof should ? require('should') : should
);
