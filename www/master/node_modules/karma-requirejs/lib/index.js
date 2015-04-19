var requirejsPath, createPattern;

createPattern = function(path) {
  return {pattern: path, included: true, served: true, watched: false};
};

requirejsPath = require('path').dirname(require.resolve('requirejs')) + '/../require.js';

var initRequireJs = function(files) {
  files.unshift(createPattern(__dirname + '/adapter.js'));
  files.unshift(createPattern(requirejsPath));
};

initRequireJs.$inject = ['config.files'];

module.exports = {
  'framework:requirejs': ['factory', initRequireJs]
};
