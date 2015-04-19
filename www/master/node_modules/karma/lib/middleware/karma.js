/**
 * Karma middleware is responsible for serving:
 * - client.html (the entrypoint for capturing a browser)
 * - debug.html
 * - context.html (the execution context, loaded within an iframe)
 * - karma.js
 *
 * The main part is generating context.html, as it contains:
 * - generating mappings
 * - including <script> and <link> tags
 * - setting propert caching headers
 */

var path = require('path');
var util = require('util');

var common = require('./common');

var VERSION = require('../constants').VERSION;
var SCRIPT_TAG = '<script type="%s" src="%s"></script>';
var LINK_TAG = '<link type="text/css" href="%s" rel="stylesheet">';
var SCRIPT_TYPE = {
  '.js': 'text/javascript',
  '.dart': 'application/dart'
};


var filePathToUrlPath = function(filePath, basePath) {
  if (filePath.indexOf(basePath) === 0) {
    return '/base' + filePath.substr(basePath.length);
  }

  return '/absolute' + filePath;
};

var createKarmaMiddleware = function(filesPromise, serveStaticFile,
    /* config.basePath */ basePath,  /* config.urlRoot */ urlRoot) {

  return function(request, response, next) {
    var requestUrl = request.url.replace(/\?.*/, '');

    // redirect /__karma__ to /__karma__ (trailing slash)
    if (requestUrl === urlRoot.substr(0, urlRoot.length - 1)) {
      response.setHeader('Location', urlRoot);
      response.writeHead(301);
      return response.end('MOVED PERMANENTLY');
    }

    // ignore urls outside urlRoot
    if (requestUrl.indexOf(urlRoot) !== 0) {
      return next();
    }

    // remove urlRoot prefix
    requestUrl = requestUrl.substr(urlRoot.length - 1);

    // serve client.html
    if (requestUrl === '/') {
      return serveStaticFile('/client.html', response);
    }

    // serve karma.js
    if (requestUrl === '/karma.js') {
      return serveStaticFile(requestUrl, response, function(data) {
        return data.replace('%KARMA_URL_ROOT%', urlRoot)
                   .replace('%KARMA_VERSION%', VERSION);
      });
    }

    // serve context.html - execution context within the iframe
    // or debug.html - execution context without channel to the server
    if (requestUrl === '/context.html' || requestUrl === '/debug.html') {
      return filesPromise.then(function(files) {
        serveStaticFile(requestUrl, response, function(data) {
          common.setNoCacheHeaders(response);

          var scriptTags = files.included.map(function(file) {
            var filePath = file.path;
            var fileExt = path.extname(filePath);

            if (!file.isUrl) {
              // TODO(vojta): serve these files from within urlRoot as well
              filePath = filePathToUrlPath(filePath, basePath);

              if (requestUrl === '/context.html') {
                filePath += '?' + file.mtime.getTime();
              }
            }

            if (fileExt === '.css') {
              return util.format(LINK_TAG, filePath);
            }

            return util.format(SCRIPT_TAG, SCRIPT_TYPE[fileExt] || 'text/javascript', filePath);
          });

          // TODO(vojta): don't compute if it's not in the template
          var mappings = files.served.map(function(file) {
            var filePath = filePathToUrlPath(file.path, basePath);

            return util.format('  \'%s\': \'%d\'', filePath, file.mtime.getTime());
          });

          mappings = 'window.__karma__.files = {\n' + mappings.join(',\n') + '\n};\n';

          return data.replace('%SCRIPTS%', scriptTags.join('\n')).replace('%MAPPINGS%', mappings);
        });
      });
    }

    return next();
  };
};


// PUBLIC API
exports.create = createKarmaMiddleware;
