var es = require('event-stream');
var html = require('html');
var clone = require('clone');

module.exports = function(options){
  var opts = options ? clone(options) : {};
  return es.map(function(file, cb){
  	file.contents = new Buffer(html.prettyPrint(String(file.contents), opts));
  	cb(null, file);
  });
}