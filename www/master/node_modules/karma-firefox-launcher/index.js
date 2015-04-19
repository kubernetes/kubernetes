var fs = require('fs');
var spawn = require('child_process').spawn;


var PREFS =
    'user_pref("browser.shell.checkDefaultBrowser", false);\n' +
    'user_pref("browser.bookmarks.restore_default_bookmarks", false);\n' +
    'user_pref("dom.disable_open_during_load", false);\n' +
    'user_pref("dom.max_script_run_time", 0);\n';


// Return location of firefox.exe file for a given Firefox directory
// (available: "Mozilla Firefox", "Aurora", "Nightly").
var getFirefoxExe = function(firefoxDirName) {
  if (process.platform !== 'win32') {
    return null;
  }


  var prefix;
  var prefixes = [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)']];
  var suffix = '\\'+ firefoxDirName + '\\firefox.exe';

  for (var i = 0; i < prefixes.length; i++) {
    prefix = prefixes[i];
    if (fs.existsSync(prefix + suffix)) {
      return prefix + suffix;
    }
  }

  return 'C:\\Program Files' + suffix;
}

// https://developer.mozilla.org/en-US/docs/Command_Line_Options
var FirefoxBrowser = function(id, baseBrowserDecorator, args, logger) {
  baseBrowserDecorator(this);

  var log = logger.create('launcher');
  this._getPrefs = function(prefs) {
    if (typeof prefs !== 'object') {
      return PREFS;
    }
    var result = PREFS;
    for (var key in prefs) {
      result += 'user_pref("' + key + '", ' + JSON.stringify(prefs[key]) + ');\n';
    }
    return result;
  }

  this._start = function(url) {
    var self = this;
    var command = this._getCommand();

    fs.writeFileSync(self._tempDir + '/prefs.js', this._getPrefs(args.prefs));
    self._execCommand(command, [url, '-profile', self._tempDir, '-no-remote']);
  };
};


FirefoxBrowser.prototype = {
  name: 'Firefox',

  DEFAULT_CMD: {
    linux: 'firefox',
    darwin: '/Applications/Firefox.app/Contents/MacOS/firefox-bin',
    win32: getFirefoxExe('Mozilla Firefox')
  },
  ENV_CMD: 'FIREFOX_BIN'
};

FirefoxBrowser.$inject = ['id', 'baseBrowserDecorator', 'args', 'logger'];


var FirefoxDeveloperBrowser = function() {
  FirefoxBrowser.apply(this, arguments);
};

FirefoxDeveloperBrowser.prototype = {
  name: 'FirefoxDeveloper',
  DEFAULT_CMD: {
    linux: 'firefox',
    darwin: '/Applications/FirefoxDeveloper.app/Contents/MacOS/firefox-bin',
    win32: getFirefoxExe('Firefox Developer Edition')
  },
  ENV_CMD: 'FIREFOX_DEVELOPER_BIN'
};

FirefoxDeveloperBrowser.$inject = ['id', 'baseBrowserDecorator', 'args', 'logger'];


var FirefoxAuroraBrowser = function() {
  FirefoxBrowser.apply(this, arguments);
};

FirefoxAuroraBrowser.prototype = {
  name: 'FirefoxAurora',
  DEFAULT_CMD: {
    linux: 'firefox',
    darwin: '/Applications/FirefoxAurora.app/Contents/MacOS/firefox-bin',
    win32: getFirefoxExe('Aurora')
  },
  ENV_CMD: 'FIREFOX_AURORA_BIN'
};

FirefoxAuroraBrowser.$inject = ['id', 'baseBrowserDecorator', 'args', 'logger'];


var FirefoxNightlyBrowser = function() {
  FirefoxBrowser.apply(this, arguments);
};

FirefoxNightlyBrowser.prototype = {
  name: 'FirefoxNightly',

  DEFAULT_CMD: {
    linux: 'firefox',
    darwin: '/Applications/FirefoxNightly.app/Contents/MacOS/firefox-bin',
    win32: getFirefoxExe('Nightly')
  },
  ENV_CMD: 'FIREFOX_NIGHTLY_BIN'
};

FirefoxNightlyBrowser.$inject = ['id', 'baseBrowserDecorator', 'args', 'logger'];


// PUBLISH DI MODULE
module.exports = {
  'launcher:Firefox': ['type', FirefoxBrowser],
  'launcher:FirefoxDeveloper': ['type', FirefoxDeveloperBrowser],
  'launcher:FirefoxAurora': ['type', FirefoxAuroraBrowser],
  'launcher:FirefoxNightly': ['type', FirefoxNightlyBrowser]
};
