
/**
 * Module dependencies.
 */

var qs = require('qs');

module.exports = function(res, fn){
  res.text = '';
  res.setEncoding('ascii');
  res.on('data', function(chunk){ res.text += chunk; });
  res.on('end', function(){
    try {
      fn(null, qs.parse(res.text));
    } catch (err) {
      fn(err);
    }
  });
};