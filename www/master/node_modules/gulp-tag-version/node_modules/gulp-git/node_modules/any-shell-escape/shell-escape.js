var util = require('util'),
    winCmd = /^win/.test(process.platform),
    escapePath;

function escapePathSh(path) {
  if (!/^[A-Za-z0-9_\/-]+$/.test(path))
    return ("'" + path.replace(/'/g, "'\"'\"'") + "'").replace(/''/g, '');
  else
    return path;
}

function escapePathWin(path) {
  if (!/^[A-Za-z0-9_\/-]+$/.test(path))
    return '"' + path.replace(/"/g, '""') + '"';
  else
    return path;
}

if (winCmd) {
  escapePath = escapePathWin;
} else {
  escapePath = escapePathSh;
}

module.exports = function(stringOrArray) {
  var ret = [];

  if (typeof(stringOrArray) == 'string') {
    return escapePath(stringOrArray);
  } else {      
    stringOrArray.forEach(function(member) {
      ret.push(escapePath(member));
    });
    return ret.join(' ');
  }
};

if (winCmd) {
  // Cannot escape messages on windows
  module.exports.msg = function(x) {
    if (typeof(x) == 'string') {
      return x;
    } else {       
      return x.join(' ');
    }
  };
} else {
  module.exports.msg = module.exports;
}
