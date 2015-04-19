module.exports = function(moduleName) {
  try {
    return module.parent.require(moduleName);
  } catch (e) {}
};
