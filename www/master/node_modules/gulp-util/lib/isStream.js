var Stream = require('stream').Stream;

module.exports = function(o) {
  return !!o && o instanceof Stream;
};
