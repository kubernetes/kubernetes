var pipeline = require('multipipe');

module.exports = function(){
  var args = arguments;
  if (args.length === 1 && Array.isArray(args[0])) {
    args = args[0];
  }
  return function(){
    return pipeline.apply(pipeline, args);
  };
};
