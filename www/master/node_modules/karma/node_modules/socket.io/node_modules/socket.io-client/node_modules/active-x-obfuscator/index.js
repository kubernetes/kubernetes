var ZeParser = require('zeparser').ZeParser;
var Util     = require('util');

module.exports = ActiveXObfuscator;
function ActiveXObfuscator(code) {
  if (!(this instanceof ActiveXObfuscator)) {
    var obfuscator = new ActiveXObfuscator(code);
    obfuscator.execute();
    return obfuscator.toString();
  }

  this._parser = ZeParser.createParser(code);
}

var OBFUSCATED_ACTIVE_X_OBJECT = ActiveXObfuscator.OBFUSCATED_ACTIVE_X_OBJECT =
  "(['Active'].concat('Object').join('X'))";
var OBFUSCATED_ACTIVE_X = ActiveXObfuscator.OBFUSCATED_ACTIVE_X =
  "(['Active'].concat('').join('X'))";

ActiveXObfuscator.prototype.execute = function() {
  this._parser.tokenizer.fixValues();
  this._obfuscate(this.getAst());
};

ActiveXObfuscator.prototype.getAst = function() {
  return this._parser.stack;
};

ActiveXObfuscator.prototype.getWhiteTokens = function() {
  return this._parser.tokenizer.wtree;
};

ActiveXObfuscator.prototype._obfuscate = function(ast) {
  var self = this;

  ast.forEach(function(node, index) {
    if (Array.isArray(node)) {
      self._obfuscate(node);
      return;
    }

    switch (node.value) {
      case 'ActiveXObject':
        if (!node.isPropertyName) {
          node.value = 'window[' + OBFUSCATED_ACTIVE_X_OBJECT + ']';
          break;
        }

        var dot = ast[index - 1]
        var whiteTokens = self.getWhiteTokens();
        whiteTokens[dot.tokposw].value = '';

        node.value = '[' + OBFUSCATED_ACTIVE_X_OBJECT + ']';
        break;
      case "'ActiveXObject'":
      case '"ActiveXObject"':
        node.value = OBFUSCATED_ACTIVE_X_OBJECT;
        break;
      case "'ActiveX'":
      case '"ActiveX"':
        node.value = OBFUSCATED_ACTIVE_X;
        break;
      default:
        if (!/ActiveX/i.test(node.value)) {
          break;
        }

        if (!node.isComment) {
          throw new Error('Unknown ActiveX occurence in: ' + Util.inspect(node));
        }

        node.value = node.value.replace(/ActiveX/i, 'Ac...eX');
    }

  });
};

ActiveXObfuscator.prototype.toString = function() {
  var whiteTokens = this.getWhiteTokens();
  return whiteTokens.reduce(function(output, node) {
    return output += node.value;
  }, '');
};
