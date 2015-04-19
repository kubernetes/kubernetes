var Tokenizer = require('./tokenizer');
var SimpleOptimizer = require('./optimizers/simple');
var AdvancedOptimizer = require('./optimizers/advanced');

function SelectorsOptimizer(options, context) {
  this.options = options || {};
  this.context = context || {};
}

SelectorsOptimizer.prototype.process = function (data, stringifier) {
  var tokens = new Tokenizer(this.context, this.options.advanced, this.options.sourceMap).toTokens(data);

  new SimpleOptimizer(this.options).optimize(tokens);
  if (this.options.advanced)
    new AdvancedOptimizer(this.options, this.context).optimize(tokens);

  return stringifier.toString(tokens);
};

module.exports = SelectorsOptimizer;
