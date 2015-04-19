module.exports = function (path) {
  try {
    return require(path);
  } catch (e) {}
};
