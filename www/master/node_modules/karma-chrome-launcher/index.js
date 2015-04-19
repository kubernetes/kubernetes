var fs = require('fs'),
  which = require('which');

var ChromeBrowser = function(baseBrowserDecorator, args) {
  baseBrowserDecorator(this);

  var flags = args.flags || [];

  this._getOptions = function(url) {
    // Chrome CLI options
    // http://peter.sh/experiments/chromium-command-line-switches/
    flags.forEach(function(flag, i) {
      if(isJSFlags(flag)) flags[i] = sanitizeJSFlags(flag);
    });

    return [
      '--user-data-dir=' + this._tempDir,
      '--no-default-browser-check',
      '--no-first-run',
      '--disable-default-apps',
      '--disable-popup-blocking',
      '--disable-translate'
    ].concat(flags, [url]);
  };
};

function isJSFlags (flag) {
  return flag.indexOf('--js-flags=') === 0;
}

function sanitizeJSFlags (flag) {
  var test = /--js-flags=(['"])/.exec(flag);
  if (!test) return flag;
  var escapeChar = test[1];
  var endExp = new RegExp(escapeChar + '$');
  var startExp = new RegExp('--js-flags=' + escapeChar);
  return flag.replace(startExp, '--js-flags=').replace(endExp, '');
}

// Return location of chrome.exe file for a given Chrome directory (available: "Chrome", "Chrome SxS").
function getChromeExe(chromeDirName) {
  // Only run these checks on win32
  if (process.platform !== 'win32') {
    return null;
  }
  var windowsChromeDirectory, i, prefix;
  var suffix = '\\Google\\'+ chromeDirName + '\\Application\\chrome.exe';
  var prefixes = [process.env.LOCALAPPDATA, process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)']];

  for (i = 0; i < prefixes.length; i++) {
    prefix = prefixes[i];
    if (fs.existsSync(prefix + suffix)) {
      windowsChromeDirectory = prefix + suffix;
      break;
    }
  }

  return windowsChromeDirectory;
}

function getBin(commands) {
  // Don't run these checks on win32
  if (process.platform !== 'linux') {
    return null;
  }
  var bin, i;
  for (i = 0; i < commands.length; i++) {
    try {
      if (which.sync(commands[i])) {
        bin = commands[i];
        break;
      }
    } catch (e) {}
  }
  return bin;
}

ChromeBrowser.prototype = {
  name: 'Chrome',

  DEFAULT_CMD: {
    // Try chromium-browser before chromium to avoid conflict with the legacy
    // chromium-bsu package previously known as 'chromium' in Debian and Ubuntu.
    linux: getBin(['chromium-browser', 'chromium', 'google-chrome', 'google-chrome-stable']),
    darwin: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    win32: getChromeExe('Chrome')
  },
  ENV_CMD: 'CHROME_BIN'
};

ChromeBrowser.$inject = ['baseBrowserDecorator', 'args'];


var ChromeCanaryBrowser = function(baseBrowserDecorator, args) {
  ChromeBrowser.call(this, baseBrowserDecorator, args);

  var parentOptions = this._getOptions;
  this._getOptions = function(url) {
    return canaryGetOptions.call(this, url, args, parentOptions);
  };
};

function canaryGetOptions (url, args, parent) {
  // disable crankshaft optimizations, as it causes lot of memory leaks (as of Chrome 23.0)
  var flags = args.flags || [];
  var augmentedFlags;
  var customFlags = '--nocrankshaft --noopt';

  flags.forEach(function(flag, i) {
    if (isJSFlags(flag)) {
      augmentedFlags = sanitizeJSFlags(flag) + ' ' + customFlags;
    }
  });

  return parent.call(this, url).concat([augmentedFlags || '--js-flags=' + customFlags]);
}

ChromeCanaryBrowser.prototype = {
  name: 'ChromeCanary',

  DEFAULT_CMD: {
    linux: 'google-chrome-canary',
    darwin: '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
    win32: getChromeExe('Chrome SxS')
  },
  ENV_CMD: 'CHROME_CANARY_BIN'
};

ChromeCanaryBrowser.$inject = ['baseBrowserDecorator', 'args'];

var DartiumBrowser = function(baseBrowserDecorator, args) {
    ChromeBrowser.call(this, baseBrowserDecorator, args);

    var checkedFlag = '--checked';
    var dartFlags = process.env['DART_FLAGS'] || '';
    var flags = dartFlags.split(' ')
    if(flags.indexOf(checkedFlag) == -1) {
        flags.push(checkedFlag);
        process.env['DART_FLAGS'] = flags.join(' ');
    }
};

DartiumBrowser.prototype = {
    name: 'Dartium',
    DEFAULT_CMD: {},
    ENV_CMD: 'DARTIUM_BIN'
};

DartiumBrowser.$inject = ['baseBrowserDecorator', 'args'];


// PUBLISH DI MODULE
module.exports = {
  'launcher:Chrome': ['type', ChromeBrowser],
  'launcher:ChromeCanary': ['type', ChromeCanaryBrowser],
  'launcher:Dartium': ['type', DartiumBrowser]
};

module.exports.test = {
  isJSFlags: isJSFlags,
  sanitizeJSFlags: sanitizeJSFlags,
  canaryGetOptions: canaryGetOptions
}