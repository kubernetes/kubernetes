function SourceTracker() {
  this.sources = [];
}

SourceTracker.prototype.store = function (filename, data) {
  this.sources.push(filename);

  return '__ESCAPED_SOURCE_CLEAN_CSS' + (this.sources.length - 1) + '__' +
    data +
    '__ESCAPED_SOURCE_END_CLEAN_CSS__';
};

SourceTracker.prototype.nextStart = function (data) {
  var next = /__ESCAPED_SOURCE_CLEAN_CSS(\d+)__/.exec(data);

  return next ?
    { index: next.index, filename: this.sources[~~next[1]] } :
    null;
};

SourceTracker.prototype.nextEnd = function (data) {
  return /__ESCAPED_SOURCE_END_CLEAN_CSS__/g.exec(data);
};

SourceTracker.prototype.removeAll = function (data) {
  return data
    .replace(/__ESCAPED_SOURCE_CLEAN_CSS\d+__/g, '')
    .replace(/__ESCAPED_SOURCE_END_CLEAN_CSS__/g, '');
};

module.exports = SourceTracker;
