var util = require('util');

var DEFAULTS = {
  '*': {
    colors: {
      opacity: true // rgba / hsla
    },
    properties: {
      backgroundSizeMerging: false, // background-size to shorthand
      iePrefixHack: false, // underscore / asterisk prefix hacks on IE
      ieSuffixHack: false, // \9 suffix hacks on IE
      merging: true // merging properties into one
    },
    selectors: {
      ie7Hack: false, // *+html hack
      special: /(\-moz\-|\-ms\-|\-o\-|\-webkit\-|:dir\([a-z-]*\)|:first(?![a-z-])|:fullscreen|:left|:read-only|:read-write|:right)/ // special selectors which prevent merging
    },
    units: {
      rem: true
    }
  },
  'ie8': {
    colors: {
      opacity: false
    },
    properties: {
      backgroundSizeMerging: false,
      iePrefixHack: true,
      ieSuffixHack: true,
      merging: false
    },
    selectors: {
      ie7Hack: false,
      special: /(\-moz\-|\-ms\-|\-o\-|\-webkit\-|:root|:nth|:first\-of|:last|:only|:empty|:target|:checked|::selection|:enabled|:disabled|:not)/
    },
    units: {
      rem: false
    }
  },
  'ie7': {
    colors: {
      opacity: false
    },
    properties: {
      backgroundSizeMerging: false,
      iePrefixHack: true,
      ieSuffixHack: true,
      merging: false
    },
    selectors: {
      ie7Hack: true,
      special: /(\-moz\-|\-ms\-|\-o\-|\-webkit\-|:focus|:before|:after|:root|:nth|:first\-of|:last|:only|:empty|:target|:checked|::selection|:enabled|:disabled|:not)/
    },
    units: {
      rem: false
    }
  }
};

function Compatibility(source) {
  this.source = source || {};
}

function merge(source, target) {
  for (var key in source) {
    var value = source[key];

    if (typeof value === 'object' && !util.isRegExp(value))
      target[key] = merge(value, target[key] || {});
    else
      target[key] = key in target ? target[key] : value;
  }

  return target;
}

function calculateSource(source) {
  if (typeof source == 'object')
    return source;

  if (!/[,\+\-]/.test(source))
    return DEFAULTS[source] || DEFAULTS['*'];

  var parts = source.split(',');
  var template = parts[0] in DEFAULTS ?
    DEFAULTS[parts.shift()] :
    DEFAULTS['*'];

  source = {};

  parts.forEach(function (part) {
    var isAdd = part[0] == '+';
    var key = part.substring(1).split('.');
    var group = key[0];
    var option = key[1];

    source[group] = source[group] || {};
    source[group][option] = isAdd;
  });

  return merge(template, source);
}

Compatibility.prototype.toOptions = function () {
  return merge(DEFAULTS['*'], calculateSource(this.source));
};

module.exports = Compatibility;
