var _ = require('underscore');
_.mixin(require('underscore.string').exports());

exports.ipv4 = function (ocets, prefix) {
  return {
    ocets: ocets,
    prefix: prefix,
    toString: function () {
      return [ocets.join('.'), prefix].join('/');
    }
  }
};

exports.hostname = function hostname (n, prefix) {
  return _.template("<%= pre %>-<%= seq %>")({
    pre: prefix || 'core',
    seq: _.pad(n, 2, '0'),
  });
};

exports.rand_string = function () {
  var crypto = require('crypto');
  var shasum = crypto.createHash('sha256');
  shasum.update(crypto.randomBytes(256));
  return shasum.digest('hex');
};


exports.rand_suffix = exports.rand_string().substring(50);

exports.join_output_file_path = function(prefix, suffix) {
  return './output/' + [prefix, exports.rand_suffix, suffix].join('_');
};
