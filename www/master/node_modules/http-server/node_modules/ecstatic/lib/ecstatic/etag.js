module.exports = function (stat) {
  return JSON.stringify([stat.ino, stat.size, stat.mtime].join('-'));
}
