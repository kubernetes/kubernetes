
/*!
 * socket.io-node
 * Copyright(c) 2011 LearnBoost <dev@learnboost.com>
 * MIT Licensed
 */

/**
 * Module dependencies.
 */

/**
 * Converts an enumerable to an array.
 *
 * @api public
 */

exports.toArray = function (enu) {
  var arr = [];

  for (var i = 0, l = enu.length; i < l; i++)
    arr.push(enu[i]);

  return arr;
};

/**
 * Unpacks a buffer to a number.
 *
 * @api public
 */

exports.unpack = function (buffer) {
  var n = 0;
  for (var i = 0; i < buffer.length; ++i) {
    n = (i == 0) ? buffer[i] : (n * 256) + buffer[i];
  }
  return n;
}

/**
 * Left pads a string.
 *
 * @api public
 */

exports.padl = function (s,n,c) { 
  return new Array(1 + n - s.length).join(c) + s;
}

