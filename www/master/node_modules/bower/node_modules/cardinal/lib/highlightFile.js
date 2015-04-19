var fs = require('fs')
  , highlight = require('./highlight');

function isFunction (obj) {
  return toString.call(obj) == '[object Function]';
}

module.exports = function highlightFile (fullPath, opts, cb) {
  if (isFunction(opts)) { 
    cb = opts;
    opts = { };
  }
  opts = opts || { };
  if (opts.json !== false && fullPath.match(/\.json$/i)) {
    opts.json = true;
  }

  fs.readFile(fullPath, 'utf-8', function (err, code) {
    if (err) return cb(err);
    try {
      cb(null, highlight(code, opts));
    } catch (e) {
      cb(e);
    }
  });
};
