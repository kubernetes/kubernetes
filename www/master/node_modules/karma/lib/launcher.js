var log = require('./logger').create('launcher');
var baseBrowserDecoratorFactory = require('./launchers/Base').decoratorFactory;


var Launcher = function(emitter, injector) {
  var browsers = [];

  this.launch = function(names, hostname, port, urlRoot) {
    var url = 'http://' + hostname + ':' + port + urlRoot;
    var browser;

    names.forEach(function(name) {
      var locals = {
        id: ['value', Launcher.generateId()],
        name: ['value', name],
        baseBrowserDecorator: ['factory', baseBrowserDecoratorFactory]
      };

      // TODO(vojta): determine script from name
      if (name.indexOf('/') !== -1) {
        name = 'Script';
      }

      try {
        browser = injector.createChild([locals], ['launcher:' + name]).get('launcher:' + name);
      } catch (e) {
        if (e.message.indexOf('No provider for "launcher:' + name + '"') !== -1) {
          log.warn('Can not load "%s", it is not registered!\n  ' +
                   'Perhaps you are missing some plugin?', name);
        } else {
          log.warn('Can not load "%s"!\n  ' + e.stack, name);
        }

        return;
      }

      log.info('Starting browser %s', browser.name);
      browser.start(url);
      browsers.push(browser);
    });
  };

  this.launch.$inject = ['config.browsers', 'config.hostname', 'config.port', 'config.urlRoot'];


  this.kill = function(callback) {
    log.debug('Disconnecting all browsers');

    var remaining = 0;
    var finish = function() {
      remaining--;
      if (!remaining && callback) {
        callback();
      }
    };

    if (!browsers.length) {
      return process.nextTick(callback);
    }

    browsers.forEach(function(browser) {
      remaining++;
      browser.kill(finish);
    });
  };


  this.areAllCaptured = function() {
    return !browsers.some(function(browser) {
      return !browser.isCaptured();
    });
  };


  this.markCaptured = function(id) {
    browsers.forEach(function(browser) {
      if (browser.id === id) {
        browser.markCaptured();
      }
    });
  };


  // register events
  emitter.on('exit', this.kill);
};

Launcher.$inject = ['emitter', 'injector'];

Launcher.generateId = function() {
  return Math.floor(Math.random() * 100000000);
};


// PUBLISH
exports.Launcher = Launcher;
