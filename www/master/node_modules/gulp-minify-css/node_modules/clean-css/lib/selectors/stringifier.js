var lineBreak = require('os').EOL;

function Stringifier(options, restoreCallback) {
  this.keepBreaks = options.keepBreaks;
  this.restoreCallback = restoreCallback;
}

function valueRebuilder(list, separator) {
  var merged = '';

  for (var i = 0, l = list.length; i < l; i++) {
    var el = list[i];

    if (el.value.indexOf('__ESCAPED_') === 0) {
      merged += el.value;

      if (i === l - 1) {
        var lastSemicolonAt = merged.lastIndexOf(';');
        merged = merged.substring(0, lastSemicolonAt) + merged.substring(lastSemicolonAt + 1);
      }
    } else {
      merged += list[i].value + (i < l - 1 ? separator : '');
    }
  }

  return merged;
}

function rebuild(tokens, keepBreaks, isFlatBlock) {
  var joinCharacter = isFlatBlock ? ';' : (keepBreaks ? lineBreak : '');
  var parts = [];
  var body;
  var selector;

  for (var i = 0, l = tokens.length; i < l; i++) {
    var token = tokens[i];

    if (token.kind === 'text' || token.kind == 'at-rule') {
      parts.push(token.value);
      continue;
    }

    // FIXME: broken due to joining/splitting
    if (token.body && (token.body.length === 0 || (token.body.length == 1 && token.body[0].value === '')))
      continue;

    if (token.kind == 'block') {
      body = token.isFlatBlock ?
        valueRebuilder(token.body, ';') :
        rebuild(token.body, keepBreaks, token.isFlatBlock);
      if (body.length > 0)
        parts.push(token.value + '{' + body + '}');
    } else {
      selector = valueRebuilder(token.value, ',');
      body = valueRebuilder(token.body, ';');
      parts.push(selector + '{' + body + '}');
    }
  }

  return parts.join(joinCharacter);
}

Stringifier.prototype.toString = function (tokens) {
  var rebuilt = rebuild(tokens, this.keepBreaks, false);

  return {
    styles: this.restoreCallback(rebuilt).trim()
  };
};

module.exports = Stringifier;
