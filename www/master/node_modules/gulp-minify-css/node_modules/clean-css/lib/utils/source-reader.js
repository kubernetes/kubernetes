var path = require('path');

function SourceReader(context, data) {
  this.outerContext = context;
  this.data = data;
}

SourceReader.prototype.toString = function () {
  if (typeof this.data == 'string')
    return this.data;
  if (Buffer.isBuffer(this.data))
    return this.data.toString();
  if (Array.isArray(this.data))
    return fromArray(this.outerContext, this.data);

  return this.data;
};

function fromArray(outerContext, sources) {
  return sources
    .map(function (source) {
      return outerContext.options.processImport === false ?
        source + '@shallow' :
        source;
    })
    .map(function (source) {
      return !outerContext.options.relativeTo || /^https?:\/\//.test(source) ?
        source :
        path.relative(outerContext.options.relativeTo, source);
    })
    .map(function (source) { return '@import url(' + source + ');'; })
    .join('');
}

module.exports = SourceReader;
