module.exports = function(module) {
  try {
    return require(module);
  } catch (e) {}
};
