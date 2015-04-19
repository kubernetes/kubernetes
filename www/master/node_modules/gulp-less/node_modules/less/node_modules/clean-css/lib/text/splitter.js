var Splitter = function Splitter (separator) {
  this.separator = separator;
};

Splitter.prototype.split = function (value) {
  if (value.indexOf(this.separator) === -1)
    return [value];

  if (value.indexOf('(') === -1)
    return value.split(this.separator);

  var level = 0;
  var cursor = 0;
  var lastStart = 0;
  var len = value.length;
  var tokens = [];

  while (cursor++ < len) {
    if (value[cursor] == '(') {
      level++;
    } else if (value[cursor] == ')') {
      level--;
    } else if (value[cursor] == this.separator && level === 0) {
      tokens.push(value.substring(lastStart, cursor));
      lastStart = cursor + 1;
    }
  }

  if (lastStart < cursor + 1)
    tokens.push(value.substring(lastStart));

  return tokens;
};

module.exports = Splitter;
