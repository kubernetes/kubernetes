var path = require('path');

module.exports = function(npath, ext) {
  var nFileName = path.basename(npath, path.extname(npath))+ext;
  return path.join(path.dirname(npath), nFileName);
};