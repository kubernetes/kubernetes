
/**
 * Module dependencies.
 */

var CookieJar = require('cookiejar').CookieJar;
var CookieAccess = require('cookiejar').CookieAccessInfo;
var parse = require('url').parse;
var request = require('./index');
var methods = require('methods');

/**
 * Expose `Agent`.
 */

module.exports = Agent;

/**
 * Initialize a new `Agent`.
 *
 * @api public
 */

function Agent() {
  if (!(this instanceof Agent)) return new Agent;
  this.jar = new CookieJar;
}

/**
 * Save the cookies in the given `res` to
 * the agent's cookie jar for persistence.
 *
 * @param {Response} res
 * @api private
 */

Agent.prototype.saveCookies = function(res){
  var cookies = res.headers['set-cookie'];
  if (cookies) this.jar.setCookies(cookies);
};

/**
 * Attach cookies when available to the given `req`.
 *
 * @param {Request} req
 * @api private
 */

Agent.prototype.attachCookies = function(req){
  var url = parse(req.url);
  var access = CookieAccess(url.host, url.pathname, 'https:' == url.protocol);
  var cookies = this.jar.getCookies(access).toValueString();
  req.cookies = cookies;
};

// generate HTTP verb methods

methods.forEach(function(method){
  var name = 'delete' == method ? 'del' : method;

  method = method.toUpperCase();
  Agent.prototype[name] = function(url, fn){
    var req = request(method, url);

    req.on('response', this.saveCookies.bind(this));
    req.on('redirect', this.saveCookies.bind(this));
    req.on('redirect', this.attachCookies.bind(this, req));
    this.attachCookies(req);

    fn && req.end(fn);
    return req;
  };
});
