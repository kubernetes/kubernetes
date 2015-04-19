
/**
 * Module dependencies.
 */

var utils = require('./utils');
var Stream = require('stream').Stream;
var mime = require('mime');
var path = require('path');
var basename = path.basename;

/**
 * Expose `Part`.
 */

module.exports = Part;

/**
 * Initialize a new `Part` for the given `req`.
 *
 * @param {Request} req
 * @api public
 */

function Part(req) {
  this.req = req;
  this.header = {};
  this.headerSent = false;
  this.request = req.request();
  this.writable = true;
  if (!req._boundary) this.assignBoundary();
}

/**
 * Inherit from `Stream.prototype`.
 */

Part.prototype.__proto__ = Stream.prototype;

/**
 * Assign the initial request-level boundary.
 *
 * @api private
 */

Part.prototype.assignBoundary = function(){
  var boundary = utils.uid(32);
  this.req.set('Content-Type', 'multipart/form-data; boundary=' + boundary);
  this.req._boundary = boundary;
};

/**
 * Set header `field` to `val`.
 *
 * @param {String} field
 * @param {String} val
 * @return {Part} for chaining
 * @api public
 */

Part.prototype.set = function(field, val){
  if (!this._boundary) {
    // TODO: formidable bug
    if (!this.request._hasFirstBoundary) {
      this.request.write('--' + this.req._boundary + '\r\n');
      this.request._hasFirstBoundary = true;
    } else {
      this.request.write('\r\n--' + this.req._boundary + '\r\n');
    }
    this._boundary = true;
  }
  this.request.write(field + ': ' + val + '\r\n');
  return this;
};

/**
 * Set _Content-Type_ response header passed through `mime.lookup()`.
 *
 * Examples:
 *
 *     res.type('html');
 *     res.type('.html');
 *
 * @param {String} type
 * @return {Part}
 * @api public
 */

Part.prototype.type = function(type){
  return this.set('Content-Type', mime.lookup(type));
};

/**
 * Set _Content-Disposition_ header field to _form-data_
 * and set the _name_ param to the given string.
 *
 * @param {String} name
 * @return {Part}
 * @api public
 */

Part.prototype.name = function(name){
  this.set('Content-Disposition', 'form-data; name="' + name + '"');
  return this;
};

/**
 * Set _Content-Disposition_ header field to _attachment_ with `filename`
 * and field `name`.
 *
 * @param {String} name
 * @param {String} filename
 * @return {Part}
 * @api public
 */

Part.prototype.attachment = function(name, filename){
  this.type(filename);
  this.set('Content-Disposition', 'attachment; name="' + name + '"; filename="' + basename(filename) + '"');
  return this;
};

/**
 * Write `data` with `encoding`.
 *
 * @param {Buffer|String} data
 * @param {String} encoding
 * @return {Boolean}
 * @api public
 */

Part.prototype.write = function(data, encoding){
  if (!this._headerCRLF) {
    this.request.write('\r\n');
    this._headerCRLF = true;
  }
  return this.request.write(data, encoding);
};

/**
 * End the part.
 *
 * @api public
 */

Part.prototype.end = function(){
  this.emit('end');
  this.complete = true;
};

