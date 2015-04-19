var io = require('socket.io');
var di = require('di');

var cfg = require('./config');
var logger = require('./logger');
var browser = require('./browser');
var constant = require('./constants');
var watcher = require('./watcher');
var plugin = require('./plugin');

var ws = require('./web-server');
var preprocessor = require('./preprocessor');
var Launcher = require('./launcher').Launcher;
var FileList = require('./file-list').List;
var reporter = require('./reporter');
var helper = require('./helper');
var events = require('./events');
var EventEmitter = events.EventEmitter;
var Executor = require('./executor');

var log = logger.create();


var start = function(injector, config, launcher, globalEmitter, preprocess, fileList, webServer,
    capturedBrowsers, socketServer, executor, done) {

  config.frameworks.forEach(function(framework) {
    injector.get('framework:' + framework);
  });

  var filesPromise = fileList.refresh();

  if (config.autoWatch) {
    filesPromise.then(function() {
      injector.invoke(watcher.watch);
    });
  }

  webServer.on('error', function(e) {
    if (e.code === 'EADDRINUSE') {
      log.warn('Port %d in use', config.port);
      config.port++;
      webServer.listen(config.port);
    } else {
      throw e;
    }
  });

  webServer.listen(config.port, function() {
    log.info('Karma v%s server started at http://%s:%s%s', constant.VERSION, config.hostname,
        config.port, config.urlRoot);

    if (config.browsers && config.browsers.length) {
      injector.invoke(launcher.launch, launcher);
    }
  });

  globalEmitter.on('browsers_change', function() {
    // TODO(vojta): send only to interested browsers
    socketServer.sockets.emit('info', capturedBrowsers.serialize());
  });

  globalEmitter.on('browser_register', function(browser) {
    // TODO(vojta): use just id
    if (browser.launchId) {
      launcher.markCaptured(browser.launchId);
    }

    // TODO(vojta): This is lame, browser can get captured and then crash (before other browsers get
    // captured).
    if ((config.autoWatch || config.singleRun) && launcher.areAllCaptured()) {
      executor.schedule();
    }
  });

  globalEmitter.on('run_complete', function(browsers, results) {
    if (config.singleRun) {
      disconnectBrowsers(results.exitCode);
    }
  });

  var EVENTS_TO_REPLY = ['start', 'info', 'error', 'result', 'complete'];
  socketServer.sockets.on('connection', function (socket) {
    log.debug('A browser has connected on socket ' + socket.id);

    var replySocketEvents = events.bufferEvents(socket, EVENTS_TO_REPLY);

    socket.on('register', function(info) {
      var newBrowser;

      if (info.id) {
        newBrowser = capturedBrowsers.getById(info.id);
      }

      if (newBrowser) {
        newBrowser.onReconnect(socket);
      } else {
        newBrowser = injector.createChild([{
          id: ['value', info.id || null],
          fullName: ['value', info.name],
          socket: ['value', socket]
        }]).instantiate(browser.Browser);

        newBrowser.init();
      }

      replySocketEvents();
    });
  });

  if (config.autoWatch) {
    globalEmitter.on('file_list_modified', function() {
      log.debug('List of files has changed, trying to execute');
      executor.schedule();
    });
  }

  var disconnectBrowsers = function(code) {
    // Slightly hacky way of removing disconnect listeners
    // to suppress "browser disconnect" warnings
    // TODO(vojta): change the client to not send the event (if disconnected by purpose)
    var sockets = socketServer.sockets.sockets;
    Object.getOwnPropertyNames(sockets).forEach(function(key) {
      sockets[key].removeAllListeners('disconnect');
    });

    globalEmitter.emitAsync('exit').then(function() {
      done(code || 0);
    });
  };


  if (config.singleRun) {
    globalEmitter.on('browser_process_failure', function(browser) {
      log.debug('%s failed to capture, aborting the run.', browser);
      disconnectBrowsers(1);
    });
  }

  try {
    process.on('SIGINT', disconnectBrowsers);
    process.on('SIGTERM', disconnectBrowsers);
  } catch (e) {
    // Windows doesn't support signals yet, so they simply don't get this handling.
    // https://github.com/joyent/node/issues/1553
  }

  // Handle all unhandled exceptions, so we don't just exit but
  // disconnect the browsers before exiting.
  process.on('uncaughtException', function(error) {
    log.error(error);
    disconnectBrowsers(1);
  });
};
start.$inject = ['injector', 'config', 'launcher', 'emitter', 'preprocess', 'fileList',
    'webServer', 'capturedBrowsers', 'socketServer', 'executor', 'done'];


var createSocketIoServer = function(webServer, executor, config) {
  var server = io.listen(webServer, {
    logger: logger.create('socket.io', constant.LOG_ERROR),
    resource: config.urlRoot + 'socket.io',
    transports: config.transports
  });

  // hack to overcome circular dependency
  executor.socketIoSockets = server.sockets;

  return server;
};


exports.start = function(cliOptions, done) {
  // apply the default logger config (and config from CLI) as soon as we can
  logger.setup(cliOptions.logLevel || constant.LOG_INFO,
      helper.isDefined(cliOptions.colors) ? cliOptions.colors : true, [constant.CONSOLE_APPENDER]);

  var config = cfg.parseConfig(cliOptions.configFile, cliOptions);
  var modules = [{
    helper: ['value', helper],
    logger: ['value', logger],
    done: ['value', done || process.exit],
    emitter: ['type', EventEmitter],
    launcher: ['type', Launcher],
    config: ['value', config],
    preprocess: ['factory', preprocessor.createPreprocessor],
    fileList: ['type', FileList],
    webServer: ['factory', ws.create],
    socketServer: ['factory', createSocketIoServer],
    executor: ['type', Executor],
    // TODO(vojta): remove
    customFileHandlers: ['value', []],
    // TODO(vojta): remove, once karma-dart does not rely on it
    customScriptTypes: ['value', []],
    reporter: ['factory', reporter.createReporters],
    capturedBrowsers: ['type', browser.Collection],
    args: ['value', {}],
    timer: ['value', {setTimeout: setTimeout, clearTimeout: clearTimeout}]
  }];

  // load the plugins
  modules = modules.concat(plugin.resolve(config.plugins));

  var injector = new di.Injector(modules);

  injector.invoke(start);
};
