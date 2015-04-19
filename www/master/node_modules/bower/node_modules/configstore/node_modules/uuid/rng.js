var rb = require('crypto').randomBytes;
module.exports = function() {
  return rb(16);
};
