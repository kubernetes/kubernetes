/*!
 * ws: a node.js websocket client
 * Copyright(c) 2011 Einar Otto Stangvik <einaros@gmail.com>
 * MIT Licensed
 */

try {
  module.exports = require('../build/Release/validation');
} catch (e) { try {
  module.exports = require('../build/default/validation');
} catch (e) { try {
  module.exports = require('./Validation.fallback');
} catch (e) {
  console.error('validation.node seems to not have been built. Run npm install.');
  throw e;
}}}
