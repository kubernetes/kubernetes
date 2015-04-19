var fs = require('fs');
var path = require('path');

var UrlRewriter = require('./url-rewriter');

function UrlRebase(outerContext) {
  this.outerContext = outerContext;
}

UrlRebase.prototype.process = function (data) {
  var options = this.outerContext.options;

  var rebaseOpts = {
    absolute: !!options.root,
    relative: !options.root && !!options.target,
    fromBase: options.relativeTo
  };

  if (!rebaseOpts.absolute && !rebaseOpts.relative)
    return data;

  if (rebaseOpts.absolute && !!options.target)
    this.outerContext.warnings.push('Both \'root\' and output file given so rebasing URLs as absolute paths');

  if (rebaseOpts.absolute)
    rebaseOpts.toBase = path.resolve(options.root);

  if (rebaseOpts.relative) {
    var target = fs.existsSync(options.target) && fs.statSync(options.target).isDirectory() ?
      options.target :
      path.dirname(options.target);

    rebaseOpts.toBase = path.resolve(target);
  }

  if (!rebaseOpts.fromBase || !rebaseOpts.toBase)
    return data;

  return new UrlRewriter(rebaseOpts).process(data);
};

module.exports = UrlRebase;
