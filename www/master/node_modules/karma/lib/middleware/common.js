/**
 * This module contains some common helpers shared between middlewares
 */

var mime = require('mime');
var log = require('../logger').create('web-server');

var PromiseContainer = function() {
  var promise;

  this.then = function(success, error) {
    return promise.then(success, error);
  };

  this.set = function(newPromise) {
    promise = newPromise;
  };
};


var serve404 = function(response, path) {
  log.warn('404: ' + path);
  response.writeHead(404);
  return response.end('NOT FOUND');
};


var createServeFile = function(fs, directory) {
  return function(filepath, response, transform) {
    if (directory) {
      filepath = directory + filepath;
    }

    return fs.readFile(filepath, function(error, data) {
      if (error) {
        return serve404(response, filepath);
      }

      response.setHeader('Content-Type', mime.lookup(filepath, 'text/plain'));

      // call custom transform fn to transform the data
      var responseData = transform && transform(data.toString()) || data;

      response.writeHead(200);

      log.debug('serving: ' + filepath);
      return response.end(responseData);
    });
  };
};


var setNoCacheHeaders = function(response) {
  response.setHeader('Cache-Control', 'no-cache');
  response.setHeader('Pragma', 'no-cache');
  response.setHeader('Expires', (new Date(0)).toString());
};


var setHeavyCacheHeaders = function(response) {
  response.setHeader('Cache-Control', ['public', 'max-age=31536000']);
};


// PUBLIC API
exports.PromiseContainer = PromiseContainer;
exports.createServeFile = createServeFile;
exports.setNoCacheHeaders = setNoCacheHeaders;
exports.setHeavyCacheHeaders = setHeavyCacheHeaders;
exports.serve404 = serve404;
