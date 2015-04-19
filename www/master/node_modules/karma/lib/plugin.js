var fs = require('fs');
var path = require('path');

var helper = require('./helper');
var log = require('./logger').create('plugin');


exports.resolve = function(plugins) {
  var modules = [];

  var requirePlugin = function(name) {
    log.debug('Loading plugin %s.', name);
    try {
      modules.push(require(name));
    } catch (e) {
      if (e.code === 'MODULE_NOT_FOUND' && e.message.indexOf(name) !== -1) {
        log.warn('Cannot find plugin "%s".\n  Did you forget to install it ?\n' +
                 '  npm install %s --save-dev', name, name);
      } else {
        log.warn('Error during loading "%s" plugin:\n  %s', name, e.message);
      }
    }
  };

  plugins.forEach(function(plugin) {
    if (helper.isString(plugin)) {
      if (plugin.indexOf('*') !== -1) {
        var pluginDirectory = path.normalize(__dirname + '/../..');
        var regexp = new RegExp('^' + plugin.replace('*', '.*'));

        log.debug('Loading %s from %s', plugin, pluginDirectory);
        fs.readdirSync(pluginDirectory).filter(function(pluginName) {
          return regexp.test(pluginName);
        }).forEach(function(pluginName) {
          requirePlugin(pluginDirectory + '/' + pluginName);
        });
      } else {
        requirePlugin(plugin);
      }
    } else if (helper.isObject(plugin)) {
      log.debug('Loading inlined plugin (defining %s).', Object.keys(plugin).join(', '));
      modules.push(plugin);
    } else {
      log.warn('Invalid plugin %s', plugin);
    }
  });

  return modules;
};
