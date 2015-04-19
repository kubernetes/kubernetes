var nock  = require('nock');
var utils = require('../../lib/utils');

var extend  = utils.extend;
var replace = utils.replace;

function Nockle(base, config) {
  if (!(this instanceof Nockle)) {
    return new Nockle(base, config);
  }

  this.base   = base;
  this.config = extend({}, config);
}

Nockle.prototype.succeed = function (method, api, values, reply) {
  values = extend({}, this.config, values);
  return nock(replace(this.base, values))[method](replace(api, values))
    .reply(200, reply || {});
};

Nockle.prototype.fail = function (method, api, values, reply) {
  values = extend({}, this.config, values);
  return nock(replace(this.base, values))[method](replace(api, values))
    .reply(404, reply || { error: 'error' });
};

Nockle.prototype.get = function (api, values, reply) {
  return this.succeed('get', api, values, reply);
};

Nockle.prototype.post = function (api, values, reply) {
  return this.succeed('post', api, values, reply);
};

Nockle.prototype.put = function (api, values, reply) {
  return this.succeed('put', api, values, reply);
};

Nockle.prototype.delete = function (api, values, reply) {
  return this.succeed('delete', api, values, reply);
};

Nockle.prototype.failget = function (api, values, reply) {
  return this.fail('get', api, values, reply);
};

Nockle.prototype.failpost = function (api, values, reply) {
  return this.fail('post', api, values, reply);
};

Nockle.prototype.failput = function (api, values, reply) {
  return this.fail('put', api, values, reply);
};

Nockle.prototype.faildelete = function (api, values, reply) {
  return this.fail('delete', api, values, reply);
};

module.exports = function (chai) {
  chai.Nockle = Nockle;
};
