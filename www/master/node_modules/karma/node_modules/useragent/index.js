'use strict';

/**
 * This is where all the magic comes from, specially crafted for `useragent`.
 */
var regexps = require('./lib/regexps');

/**
 * Reduce references by storing the lookups.
 */
// OperatingSystem parsers:
var osparsers = regexps.os
  , osparserslength = osparsers.length;

// UserAgent parsers:
var agentparsers = regexps.browser
  , agentparserslength = agentparsers.length;

// Device parsers:
var deviceparsers = regexps.device
  , deviceparserslength = deviceparsers.length;

/**
 * The representation of a parsed user agent.
 *
 * @constructor
 * @param {String} family The name of the browser
 * @param {String} major Major version of the browser
 * @param {String} minor Minor version of the browser
 * @param {String} patch Patch version of the browser
 * @param {String} source The actual user agent string
 * @api public
 */
function Agent(family, major, minor, patch, source) {
  this.family = family || 'Other';
  this.major = major || '0';
  this.minor = minor || '0';
  this.patch = patch || '0';
  this.source = source || '';
}

/**
 * OnDemand parsing of the Operating System.
 *
 * @type {OperatingSystem}
 * @api public
 */
Object.defineProperty(Agent.prototype, 'os', {
  get: function lazyparse() {
    var userAgent = this.source
      , length = osparserslength
      , parsers = osparsers
      , i = 0
      , parser
      , res;

    for (; i < length; i++) {
      if (res = parsers[i][0].exec(userAgent)) {
        parser = parsers[i];

        if (parser[1]) res[1] = parser[1].replace('$1', res[1]);
        break;
      }
    }

    return Object.defineProperty(this, 'os', {
        value: !parser || !res
          ? new OperatingSystem()
          : new OperatingSystem(
                res[1]
              , parser[2] || res[2]
              , parser[3] || res[3]
              , parser[4] || res[4]
            )
    }).os;
  },

  /**
   * Bypass the OnDemand parsing and set an OperatingSystem instance.
   *
   * @param {OperatingSystem} os
   * @api public
   */
  set: function set(os) {
    if (!(os instanceof OperatingSystem)) return false;

    return Object.defineProperty(this, 'os', {
      value: os
    }).os;
  }
});

/**
 * OnDemand parsing of the Device type.
 *
 * @type {Device}
 * @api public
 */
Object.defineProperty(Agent.prototype, 'device', {
  get: function lazyparse() {
    var userAgent = this.source
      , length = deviceparserslength
      , parsers = deviceparsers
      , i = 0
      , parser
      , res;

    for (; i < length; i++) {
      if (res = parsers[i][0].exec(userAgent)) {
        parser = parsers[i];

        if (parser[1]) res[1] = parser[1].replace('$1', res[1]);
        break;
      }
    }

    return Object.defineProperty(this, 'device', {
        value: !parser || !res
          ? new Device()
          : new Device(
                res[1]
              , parser[2] || res[2]
              , parser[3] || res[3]
              , parser[4] || res[4]
            )
    }).device;
  },

  /**
   * Bypass the OnDemand parsing and set an Device instance.
   *
   * @param {Device} device
   * @api public
   */
  set: function set(device) {
    if (!(device instanceof Device)) return false;

    return Object.defineProperty(this, 'device', {
      value: device
    }).device;
  }
});
/*** Generates a string output of the parsed user agent.
 *
 * @returns {String}
 * @api public
 */
Agent.prototype.toAgent = function toAgent() {
  var output = this.family
    , version = this.toVersion();

  if (version) output += ' '+ version;
  return output;
};

/**
 * Generates a string output of the parser user agent and operating system.
 *
 * @returns {String}  "UserAgent 0.0.0 / OS"
 * @api public
 */
Agent.prototype.toString = function toString() {
  var agent = this.toAgent()
    , os = this.os !== 'Other' ? this.os : false;

  return agent + (os ? ' / ' + os : '');
};

/**
 * Outputs a compiled veersion number of the user agent.
 *
 * @returns {String}
 * @api public
 */
Agent.prototype.toVersion = function toVersion() {
  var version = '';

  if (this.major) {
    version += this.major;

    if (this.minor) {
     version += '.' + this.minor;

     // Special case here, the patch can also be Alpha, Beta etc so we need
     // to check if it's a string or not.
     if (this.patch) {
      version += (isNaN(+this.patch) ? ' ' : '.') + this.patch;
     }
    }
  }

  return version;
};

/**
 * Outputs a JSON string of the Agent.
 *
 * @returns {String}
 * @api public
 */
Agent.prototype.toJSON = function toJSON() {
  return {
      family: this.family
    , major: this.major
    , minor: this.minor
    , patch: this.patch
    , device: this.device
    , os: this.os
  };
};

/**
 * The representation of a parsed Operating System.
 *
 * @constructor
 * @param {String} family The name of the os
 * @param {String} major Major version of the os
 * @param {String} minor Minor version of the os
 * @param {String} patch Patch version of the os
 * @api public
 */
function OperatingSystem(family, major, minor, patch) {
  this.family = family || 'Other';
  this.major = major || '';
  this.minor = minor || '';
  this.patch = patch || '';
}

/**
 * Generates a stringified version of the Operating System.
 *
 * @returns {String} "Operating System 0.0.0"
 * @api public
 */
OperatingSystem.prototype.toString = function toString() {
  var output = this.family
    , version = this.toVersion();

  if (version) output += ' '+ version;
  return output;
};

/**
 * Generates the version of the Operating System.
 *
 * @returns {String}
 * @api public
 */
OperatingSystem.prototype.toVersion = function toVersion() {
  var version = '';

  if (this.major) {
    version += this.major;

    if (this.minor) {
     version += '.' + this.minor;

     // Special case here, the patch can also be Alpha, Beta etc so we need
     // to check if it's a string or not.
     if (this.patch) {
      version += (isNaN(+this.patch) ? ' ' : '.') + this.patch;
     }
    }
  }

  return version;
};

/**
 * Outputs a JSON string of the OS, values are defaulted to undefined so they
 * are not outputed in the stringify.
 *
 * @returns {String}
 * @api public
 */
OperatingSystem.prototype.toJSON = function toJSON(){
  return {
      family: this.family
    , major: this.major || undefined
    , minor: this.minor || undefined
    , patch: this.patch || undefined
  };
};

/**
 * The representation of a parsed Device.
 *
 * @constructor
 * @param {String} family The name of the device
 * @param {String} major Major version of the device
 * @param {String} minor Minor version of the device
 * @param {String} patch Patch version of the device
 * @api public
 */
function Device(family, major, minor, patch) {
  this.family = family || 'Other';
  this.major = major || '';
  this.minor = minor || '';
  this.patch = patch || '';
}

/**
 * Generates a stringified version of the Device.
 *
 * @returns {String} "Device 0.0.0"
 * @api public
 */
Device.prototype.toString = function toString() {
  var output = this.family
    , version = this.toVersion();

  if (version) output += ' '+ version;
  return output;
};

/**
 * Generates the version of the Device.
 *
 * @returns {String}
 * @api public
 */
Device.prototype.toVersion = function toVersion() {
  var version = '';

  if (this.major) {
    version += this.major;

    if (this.minor) {
     version += '.' + this.minor;

     // Special case here, the patch can also be Alpha, Beta etc so we need
     // to check if it's a string or not.
     if (this.patch) {
      version += (isNaN(+this.patch) ? ' ' : '.') + this.patch;
     }
    }
  }

  return version;
};

/**
 * Get string representation.
 *
 * @returns {String}
 * @api public
 */
Device.prototype.toString = function toString() {
  var output = this.family
    , version = this.toVersion();

  if (version) output += ' '+ version;
  return output;
};

/**
 * Outputs a JSON string of the Device, values are defaulted to undefined so they
 * are not outputed in the stringify.
 *
 * @returns {String}
 * @api public
 */
Device.prototype.toJSON = function toJSON() {
  return {
      family: this.family
    , major: this.major || undefined
    , minor: this.minor || undefined
    , patch: this.patch || undefined
  };
};

/**
 * Small nifty thick that allows us to download a fresh set regexs from t3h
 * Int3rNetz when we want to. We will be using the compiled version by default
 * but users can opt-in for updates.
 *
 * @param {Boolean} refresh Refresh the dataset from the remote
 * @api public
 */
module.exports = function updater() {
  try {
    require('./lib/update').update(function updating(err, results) {
      if (err) {
        console.log('[useragent] Failed to update the parsed due to an error:');
        console.log('[useragent] '+ (err.message ? err.message : err));
        return;
      }

      regexps = results;

      // OperatingSystem parsers:
      osparsers = regexps.os;
      osparserslength = osparsers.length;

      // UserAgent parsers:
      agentparsers = regexps.browser;
      agentparserslength = agentparsers.length;

      // Device parsers:
      deviceparsers = regexps.device;
      deviceparserslength = deviceparsers.length;
    });
  } catch (e) {
    console.error('[useragent] If you want to use automatic updating, please add:');
    console.error('[useragent]   - request (npm install request --save)');
    console.error('[useragent]   - yamlparser (npm install yamlparser --save)');
    console.error('[useragent] To your own package.json');
  }
};

// Override the exports with our newly set module.exports
exports = module.exports;

/**
 * Nao that we have setup all the different classes and configured it we can
 * actually start assembling and exposing everything.
 */
exports.Device = Device;
exports.OperatingSystem = OperatingSystem;
exports.Agent = Agent;

/**
 * Parses the user agent string with the generated parsers from the
 * ua-parser project on google code.
 *
 * @param {String} userAgent The user agent string
 * @param {String} jsAgent Optional UA from js to detect chrome frame
 * @returns {Agent}
 * @api public
 */
exports.parse = function parse(userAgent, jsAgent) {
  if (!userAgent) return new Agent();

  var length = agentparserslength
    , parsers = agentparsers
    , i = 0
    , parser
    , res;

  for (; i < length; i++) {
    if (res = parsers[i][0].exec(userAgent)) {
      parser = parsers[i];

      if (parser[1]) res[1] = parser[1].replace('$1', res[1]);
      if (!jsAgent) return new Agent(
          res[1]
        , parser[2] || res[2]
        , parser[3] || res[3]
        , parser[4] || res[4]
        , userAgent
      );

      break;
    }
  }

  // Return early if we didn't find an match, but might still be able to parse
  // the os and device, so make sure we supply it with the source
  if (!parser || !res) return new Agent('', '', '', '', userAgent);

  // Detect Chrome Frame, but make sure it's enabled! So we need to check for
  // the Chrome/ so we know that it's actually using Chrome under the hood.
  if (jsAgent && ~jsAgent.indexOf('Chrome/') && ~userAgent.indexOf('chromeframe')) {
    res[1] = 'Chrome Frame (IE '+ res[1] +'.'+ res[2] +')';

    // Run the JavaScripted userAgent string through the parser again so we can
    // update the version numbers;
    parser = parse(jsAgent);
    parser[2] = parser.major;
    parser[3] = parser.minor;
    parser[4] = parser.patch;
  }

  return new Agent(
      res[1]
    , parser[2] || res[2]
    , parser[3] || res[3]
    , parser[4] || res[4]
    , userAgent
  );
};

/**
 * If you are doing a lot of lookups you might want to cache the results of the
 * parsed user agent string instead, in memory.
 *
 * @TODO We probably want to create 2 dictionary's here 1 for the Agent
 * instances and one for the userAgent instance mapping so we can re-use simular
 * Agent instance and lower our memory consumption.
 *
 * @param {String} userAgent The user agent string
 * @param {String} jsAgent Optional UA from js to detect chrome frame
 * @api public
 */
var LRU = require('lru-cache')(5000);
exports.lookup = function lookup(userAgent, jsAgent) {
  var key = (userAgent || '')+(jsAgent || '')
    , cached = LRU.get(key);

  if (cached) return cached;
  LRU.set(key, (cached = exports.parse(userAgent, jsAgent)));

  return cached;
};

/**
 * Does a more inaccurate but more common check for useragents identification.
 * The version detection is from the jQuery.com library and is licensed under
 * MIT.
 *
 * @param {String} useragent The user agent
 * @returns {Object} matches
 * @api public
 */
exports.is = function is(useragent) {
  var ua = (useragent || '').toLowerCase()
    , details = {
        chrome: false
      , firefox: false
      , ie: false
      , mobile_safari: false
      , mozilla: false
      , opera: false
      , safari: false
      , webkit: false
      , android: false
      , version: (ua.match(exports.is.versionRE) || [0, "0"])[1]
    };

  if (~ua.indexOf('webkit')) {
    details.webkit = true;

    if (~ua.indexOf('android')){
      details.android = true;
    }

    if (~ua.indexOf('chrome')) {
      details.chrome = true;
    } else if (~ua.indexOf('safari')) {
      details.safari = true;

      if (~ua.indexOf('mobile') && ~ua.indexOf('apple')) {
        details.mobile_safari = true;
      }
    }
  } else if (~ua.indexOf('opera')) {
    details.opera = true;
  } else if (~ua.indexOf('trident')) {
    details.ie = true;
  } else if (~ua.indexOf('mozilla') && !~ua.indexOf('compatible')) {
    details.mozilla = true;

    if (~ua.indexOf('firefox')) details.firefox = true;
  } 


  return details;
};

/**
 * Parses out the version numbers.
 *
 * @type {RegExp}
 * @api private
 */
exports.is.versionRE = /.+(?:rv|it|ra|ie)[\/: ]([\d.]+)/;

/**
 * Transform a JSON object back to a valid userAgent string
 *
 * @param {Object} details
 * @returns {Agent}
 */
exports.fromJSON = function fromJSON(details) {
  if (typeof details === 'string') details = JSON.parse(details);

  var agent = new Agent(details.family, details.major, details.minor, details.patch)
    , os = details.os;

  // The device family was added in v2.0
  if ('device' in details) {
    agent.device = new Device(details.device.family);
  } else {
    agent.device = new Device();
  }

  if ('os' in details && os) {
    // In v1.1.0 we only parsed out the Operating System name, not the full
    // version which we added in v2.0. To provide backwards compatible we should
    // we should set the details.os as family
    if (typeof os === 'string') {
      agent.os = new OperatingSystem(os);
    } else {
      agent.os = new OperatingSystem(os.family, os.major, os.minor, os.patch);
    }
  }

  return agent;
};

/**
 * Library version.
 *
 * @type {String}
 * @api public
 */
exports.version = require('./package.json').version;
