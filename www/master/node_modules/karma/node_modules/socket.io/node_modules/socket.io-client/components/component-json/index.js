
module.exports = 'undefined' == typeof JSON
  ? require('json-fallback')
  : JSON;
