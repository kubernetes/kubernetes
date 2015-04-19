var fs = require('fs');
var path = require('path');

function serializeOption(value) {
  if (typeof value === 'function') {
    return value.toString();
  }
  return JSON.stringify(value);
}

var phantomJSExePath = function () {
  // If the path we're given by phantomjs is to a .cmd, it is pointing to a global copy. 
  // Using the cmd as the process to execute causes problems cleaning up the processes 
  // so we walk from the cmd to the phantomjs.exe and use that instead.

  var phantomSource = require('phantomjs').path;

  if (path.extname(phantomSource).toLowerCase() === '.cmd') {
    return path.join(path.dirname( phantomSource ), '//node_modules//phantomjs//lib//phantom//phantomjs.exe');
  }

  return phantomSource;
};

var PhantomJSBrowser = function(baseBrowserDecorator, config, args) {
  baseBrowserDecorator(this);

  var options = args && args.options || config && config.options || {};
  var flags = args && args.flags || config && config.flags || [];

  this._start = function(url) {
    // create the js file that will open karma
    var captureFile = this._tempDir + '/capture.js';
    var optionsCode = Object.keys(options).map(function (key) {
      if (key !== 'settings') { // settings cannot be overriden, it should be extended!
        return 'page.' + key + ' = ' + serializeOption(options[key]) + ';';
      }
    });

    if (options.settings) {
      optionsCode = optionsCode.concat(Object.keys(options.settings).map(function (key) {
        return 'page.settings.' + key + ' = ' + serializeOption(options.settings[key]) + ';';
      }));
    }

    var captureCode = 'var page = require("webpage").create();\n' +
        optionsCode.join('\n') + '\npage.open("' + url + '");\n';
    fs.writeFileSync(captureFile, captureCode);

    flags = flags.concat(captureFile);

    // and start phantomjs
    this._execCommand(this._getCommand(), flags);
  };
};

PhantomJSBrowser.prototype = {
  name: 'PhantomJS',

  DEFAULT_CMD: {
    linux: require('phantomjs').path,
    darwin: require('phantomjs').path,
    win32: phantomJSExePath()
  },
  ENV_CMD: 'PHANTOMJS_BIN'
};

PhantomJSBrowser.$inject = ['baseBrowserDecorator', 'config.phantomjsLauncher', 'args'];


// PUBLISH DI MODULE
module.exports = {
  'launcher:PhantomJS': ['type', PhantomJSBrowser]
};
