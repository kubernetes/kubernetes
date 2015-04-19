const findup = require('findup-sync');

module.exports = function (search, paths) {
  var path;
  var len = paths.length;
  for (var i=0; i < len; i++) {
    if (path) {
      break;
    } else {
      path = findup(search, {cwd: paths[i], nocase: true});
    }
  }
  return path;
};
