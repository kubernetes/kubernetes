var fs = require('fs')
  , highlight = require('./highlight');

module.exports = function highlightFileSync (fullPath, opts) {
  var code = fs.readFileSync(fullPath, 'utf-8');
  opts = opts || { };
  if (opts.json !== false && fullPath.match(/\.json$/i)) {
    opts.json = true;
  }
  return highlight(code, opts);
};
