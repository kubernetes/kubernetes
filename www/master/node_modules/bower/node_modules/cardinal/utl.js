var util = require('util');

module.exports.isPath = function (s) {
  return (/[\/\\]/).test(s);
};

module.exports.inspect = function(obj, depth) {
  console.log(util.inspect(obj, false, depth || 5, true));
};


