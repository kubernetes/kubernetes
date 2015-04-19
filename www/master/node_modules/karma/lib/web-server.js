var fs = require('fs');
var http = require('http');
var path = require('path');
var connect = require('connect');

var common = require('./middleware/common');
var runnerMiddleware = require('./middleware/runner');
var karmaMiddleware = require('./middleware/karma');
var sourceFilesMiddleware = require('./middleware/source-files');
var proxyMiddleware = require('./middleware/proxy');


var createCustomHandler = function(customFileHandlers, /* config.basePath */ basePath) {
  return function(request, response, next) {
    for (var i = 0; i < customFileHandlers.length; i++) {
      if (customFileHandlers[i].urlRegex.test(request.url)) {
        return customFileHandlers[i].handler(request, response, 'fake/static', 'fake/adapter',
            basePath, 'fake/root');
      }
    }

    return next();
  };
};


var createWebServer = function(injector, emitter) {
  var serveStaticFile = common.createServeFile(fs, path.normalize(__dirname + '/../static'));
  var serveFile = common.createServeFile(fs);
  var filesPromise = new common.PromiseContainer();

  emitter.on('file_list_modified', function(files) {
    filesPromise.set(files);
  });

  // locals for webserver module
  // NOTE(vojta): figure out how to do this with DI
  injector = injector.createChild([{
    serveFile: ['value', serveFile],
    serveStaticFile: ['value', serveStaticFile],
    filesPromise: ['value', filesPromise]
  }]);

  // TODO(vojta): remove if https://github.com/senchalabs/connect/pull/850 gets merged
  var compressOptions = {
    filter: function(req, res){
      return (/json|text|javascript|dart/).test(res.getHeader('Content-Type'));
    }
  };

  var handler = connect()
      .use(connect.compress(compressOptions))
      .use(injector.invoke(runnerMiddleware.create))
      .use(injector.invoke(karmaMiddleware.create))
      .use(injector.invoke(sourceFilesMiddleware.create))
      // TODO(vojta): extract the proxy into a plugin
      .use(injector.invoke(proxyMiddleware.create))
      // TODO(vojta): remove, this is only here because of karma-dart
      // we need a better way of custom handlers
      .use(injector.invoke(createCustomHandler))
      .use(function(request, response) {
        common.serve404(response, request.url);
      });

  return http.createServer(handler);
};


// PUBLIC API
exports.create = createWebServer;
