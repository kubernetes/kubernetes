var path = require('path');

var UrlRewriter = require('./url-rewriter');

module.exports = function UrlRebase(options, context) {
  var process = function(data) {
    var rebaseOpts = {
      absolute: !!options.root,
      relative: !options.root && !!options.target,
      fromBase: options.relativeTo
    };

    if (!rebaseOpts.absolute && !rebaseOpts.relative)
      return data;

    if (rebaseOpts.absolute && !!options.target)
      context.warnings.push('Both \'root\' and output file given so rebasing URLs as absolute paths');

    if (rebaseOpts.absolute)
      rebaseOpts.toBase = path.resolve(options.root);

    if (rebaseOpts.relative)
      rebaseOpts.toBase = path.resolve(path.dirname(options.target));

    if (!rebaseOpts.fromBase || !rebaseOpts.toBase)
      return data;

    return UrlRewriter.process(data, rebaseOpts);
  };

  return { process: process };
};
