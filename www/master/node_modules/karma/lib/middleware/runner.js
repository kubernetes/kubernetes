/**
 * Runner middleware is reponsible for communication with `karma run`.
 *
 * It basically triggers a test run and streams stdout back.
 */

var path = require('path');
var helper = require('../helper');
var log = require('../logger').create();
var constant = require('../constants');
var json = require('connect').json();

// TODO(vojta): disable when single-run mode
var createRunnerMiddleware = function(emitter, fileList, capturedBrowsers, reporter, executor,
    /* config.hostname */ hostname, /* config.port */ port, /* config.urlRoot */ urlRoot, config) {

  return function(request, response, next) {

    if (request.url !== '/__run__' && request.url !== urlRoot + 'run') {
      return next();
    }

    log.debug('Execution (fired by runner)');
    response.writeHead(200);

    if (!capturedBrowsers.length) {
      var url = 'http://' + hostname + ':' + port + urlRoot;

      return response.end('No captured browser, open ' + url + '\n');
    }

    json(request, response, function() {
      if (!capturedBrowsers.areAllReady([])) {
        response.write('Waiting for previous execution...\n');
      }

      emitter.once('run_start', function() {
        var responseWrite = response.write.bind(response);

        reporter.addAdapter(responseWrite);

        // clean up, close runner response
        emitter.once('run_complete', function(browsers, results) {
          reporter.removeAdapter(responseWrite);
          response.end(constant.EXIT_CODE + results.exitCode);
        });
      });

      var data = request.body;
      log.debug('Setting client.args to ', data.args);
      config.client.args = data.args;

      var fullRefresh = true;

      if (helper.isArray(data.changedFiles)) {
        data.changedFiles.forEach(function(filepath) {
          fileList.changeFile(path.resolve(config.basePath, filepath));
          fullRefresh = false;
        });
      }

      if (helper.isArray(data.addedFiles)) {
        data.addedFiles.forEach(function(filepath) {
          fileList.addFile(path.resolve(config.basePath, filepath));
          fullRefresh = false;
        });
      }

      if (helper.isArray(data.removedFiles)) {
        data.removedFiles.forEach(function(filepath) {
          fileList.removeFile(path.resolve(config.basePath, filepath));
          fullRefresh = false;
        });
      }

      if (fullRefresh && data.refresh !== false) {
        log.debug('Refreshing all the files / patterns');
        fileList.refresh();

        if (!config.autoWatch) {
          executor.schedule();
        }
      } else {
        executor.schedule();
      }
    });
  };
};


// PUBLIC API
exports.create = createRunnerMiddleware;
