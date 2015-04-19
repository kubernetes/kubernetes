var placeholderBrace = '__';

function EscapeStore(placeholderRoot) {
  this.placeholderRoot = 'ESCAPED_' + placeholderRoot + '_CLEAN_CSS';
  this.placeholderToData = {};
  this.dataToPlaceholder = {};
  this.count = 0;
  this.restoreMatcher = new RegExp(this.placeholderRoot + '(\\d+)');
}

EscapeStore.prototype._nextPlaceholder = function (metadata) {
  return {
    index: this.count,
    value: placeholderBrace + this.placeholderRoot + this.count++ + metadata + placeholderBrace
  };
};

EscapeStore.prototype.store = function (data, metadata) {
  var encodedMetadata = metadata ?
    '(' + metadata.join(',') + ')' :
    '';
  var placeholder = this.dataToPlaceholder[data];

  if (!placeholder) {
    var nextPlaceholder = this._nextPlaceholder(encodedMetadata);
    placeholder = nextPlaceholder.value;
    this.placeholderToData[nextPlaceholder.index] = data;
    this.dataToPlaceholder[data] = nextPlaceholder.value;
  }

  if (metadata)
    placeholder = placeholder.replace(/\([^\)]+\)/, encodedMetadata);

  return placeholder;
};

EscapeStore.prototype.nextMatch = function (data, cursor) {
  var next = {};

  next.start = data.indexOf(this.placeholderRoot, cursor) - placeholderBrace.length;
  next.end = data.indexOf(placeholderBrace, next.start + placeholderBrace.length) + placeholderBrace.length;
  if (next.start > -1 && next.end > -1)
    next.match = data.substring(next.start, next.end);

  return next;
};

EscapeStore.prototype.restore = function (placeholder) {
  var index = this.restoreMatcher.exec(placeholder)[1];
  return this.placeholderToData[index];
};

module.exports = EscapeStore;
