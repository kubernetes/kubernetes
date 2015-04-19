
/**
 * Export websocket versions.
 */

module.exports = {
  7: require('./hybi-07-12'),
  8: require('./hybi-07-12'),
  13: require('./hybi-16'),
  default: require('./default')
};
