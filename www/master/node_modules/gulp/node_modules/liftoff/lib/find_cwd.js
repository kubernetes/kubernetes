const path = require('path');

module.exports = function (opts) {
  opts = opts||{};
  var cwd = opts.cwd;
  var configPath = opts.configPath;
  // if a path to the desired config was specified
  // but no cwd was provided, use configPath dir
  if (typeof configPath === 'string' && !cwd) {
    cwd = path.dirname(path.resolve(configPath));
  }
  if (typeof cwd === 'string') {
    return path.resolve(cwd);
  }
  return process.cwd();
};
