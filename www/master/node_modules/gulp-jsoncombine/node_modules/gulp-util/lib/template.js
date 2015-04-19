var template = require('lodash.template');
var reInterpolate = require('lodash._reinterpolate');

var forcedSettings = {
  escape: /<%-([\s\S]+?)%>/g,
  evaluate: /<%([\s\S]+?)%>/g,
  interpolate: reInterpolate
};

module.exports = function(tmpl, data){
  var fn = template(tmpl, null, forcedSettings);

  var wrapped = function(o) {
    if (typeof o === 'undefined' || typeof o.file === 'undefined') throw new Error('Failed to provide the current file as "file" to the template');
    return fn(o);
  };

  return (data ? wrapped(data) : wrapped);
};
