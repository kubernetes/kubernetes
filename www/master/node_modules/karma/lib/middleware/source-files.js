/**
 * Source Files middleware is responsible for serving all the source files under the test.
 */

var querystring = require('querystring');
var common = require('./common');
var pause = require('connect').utils.pause;


var findByPath = function(files, path) {
  for (var i = 0; i < files.length; i++) {
    if (files[i].path === path) {
      return files[i];
    }
  }

  return null;
};


var createSourceFilesMiddleware = function(filesPromise, serveFile,
    /* config.basePath */ basePath) {

  return function(request, response, next) {
    var requestedFilePath = querystring.unescape(request.url)
        .replace(/\?.*/, '')
        .replace(/^\/absolute/, '')
        .replace(/^\/base/, basePath);

    // Need to pause the request because of proxying, see:
    // https://groups.google.com/forum/#!topic/q-continuum/xr8znxc_K5E/discussion
    // TODO(vojta): remove once we don't care about Node 0.8
    var pausedRequest = pause(request);

    return filesPromise.then(function(files) {
      // TODO(vojta): change served to be a map rather then an array
      var file = findByPath(files.served, requestedFilePath);

      if (file) {
        serveFile(file.contentPath, response, function() {
          if (/\?\d+/.test(request.url)) {
            // files with timestamps - cache one year, rely on timestamps
            common.setHeavyCacheHeaders(response);
          } else {
            // without timestamps - no cache (debug)
            common.setNoCacheHeaders(response);
          }
        });
      } else {
        next();
      }

      pausedRequest.resume();
    });
  };
};


// PUBLIC API
exports.create = createSourceFilesMiddleware;
