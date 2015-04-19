module.exports = function (opts) {
  opts = opts || {};
  var configName = opts.configName;
  var extensions = opts.extensions;
  if (!configName) {
    throw new Error('Please specify a configName.');
  }
  if (configName instanceof RegExp) {
    return [configName];
  }
  if (!Array.isArray(extensions)) {
    throw new Error('Please provide an array of valid extensions.');
  }
  return extensions.map(function (ext) {
    return configName+ext;
  });
};
