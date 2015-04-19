/*!
 * ws: a node.js websocket client
 * Copyright(c) 2011 Einar Otto Stangvik <einaros@gmail.com>
 * MIT Licensed
 */

try {
  module.exports = require('../build/Release/bufferutil');
} catch (e) { try {
  module.exports = require('../build/default/bufferutil');
} catch (e) { try {
  module.exports = require('./BufferUtil.fallback');
} catch (e) {
  console.error('bufferutil.node seems to not have been built. Run npm install.');
  throw e;
}}}
