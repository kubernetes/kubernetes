var q = require('q'),
    fs = require('fs'),
    path = require('path'),
    SauceLabs = require('saucelabs'),
    https = require('https');

var SAUCE_LOGS_WAIT = 5000;

/**
 * Outputs information about where your Protractor test is spending its time
 * to the specified folder. A JSON data file and small index.html to view
 * it will be created. The page uses Google Charts to show the timeline.
 *
 * You enable this plugin in your config file:
 *
 *    exports.config = {
 *      plugins: [{
 *        path: 'node_modules/protractor/plugins/timeline',
 *
 *        // Output json and html will go in this folder. Relative
 *        // to current working directory of the process.
 *        // TODO - it would make more sense for this to be relative
 *        //        to the config file - reconsider this setup
 *        outdir: 'timelines',
 *
 *        // Optional - if sauceUser and sauceKey are specified, logs from
 *        // SauceLabs will also be parsed after test invocation.
 *        sauceUser: 'Jane',
 *        sauceKey: 'abcdefg'
 *      }]
 *    };
 *
 * The plugin will create timeline entries from
 *  - The Protractor test process itself.
 *  - The WebDriver Selenium Server (these logs are unavailable for Internet
 *    Explorer and for Chrome test run over Sauce Labs).
 *  - Sauce Labs job logs, if sauceUser and sauceKey are specified.
 *
 * @constructor
 */
var TimelinePlugin = function() {
  // Timelines are of the format:
  // Array<{
  //   source: string,
  //   id: number,
  //   command: string,
  //   start: number,
  //   end: number
  // }>
  this.timeline = [];

  this.clientLogAvailable = false;
  this.outdir;
  this.sessionId;
  this.testProcessSetTimeoutTimestamp = 0;
};

/**
 * Parse a selenium log in array form. For example, the logs returned
 * from the selenium standalone server are returned as arrays.
 *
 * @param {Array<Object>} logArr The selenium server logs.
 * @param {string} sourceName Descripton of source.
 * @param {number} referenceStart Date in millis.
 */
TimelinePlugin.parseArrayLog = function(logArr, sourceName, referenceStart) {
  return TimelinePlugin.parseLog(logArr, sourceName, {
    isEventStart: function(event) {
      return /Executing:/.test(event.message);
    },
    isEventEnd: function(event) {
      return /Done:/.test(event.message);
    },
    extractCommand: function(event) {
      // Messages from the Selenium Standalone server are of the form
      // org...DriverServlet Executing: [command: details [params]] at URL /url/
      return /Executing: \[([^:^\]]*)/.exec(event.message)[1];
    },
    extractTimestamp: function(event) {
      return event.timestamp;
    }
  }, referenceStart);
};

/**
 * Parse a selenium log from a string. For example, the logs returned from
 * Sauce Labs are available only as plain text.
 *
 * @param {string} text The text logs.
 * @param {string} sourceName Descripton of source.
 * @param {number} referenceStart Date in millis.
 */
TimelinePlugin.parseTextLog = function(text, sourceName, referenceStart) {
  var logLines = text.split('\n');
  var actions;

  // Look for 'standalone server' in the first couple lines of the log.
  if (/standalone server/.test(logLines.slice(0, 3).join(' '))) {
    // This is a Selenium Standalone Server log.
    actions = {
      isEventStart: function(event) {
        return /INFO - Executing:/.test(event);
      },
      isEventEnd: function(event) {
        return /INFO - Done:/.test(event);
      },
      extractCommand: function(event) {
        // Messages are of the form
        // timestamp INFO - Executing: [command: details; [params]]
        return /Executing: \[([^:^\]]*)/.exec(event)[1];
      },
      extractTimestamp: function(event) {
        // Timestamps begin the line and are formatted as
        // HH:MM:SS.SSS
        // We don't care about the date so just set it to 0.
        return Date.parse('01 Jan 1970 ' + event.slice(0, 12));
      }
    };
  } else {
    // This is a ChromeDriver log.
    actions = {
      isEventStart: function(event) {
        return /: COMMAND/.test(event);
      },
      isEventEnd: function(event) {
        return /: RESPONSE/.test(event);
      },
      extractCommand: function(event) {
        return /: COMMAND ([^\s]*)/.exec(event)[1];
      },
      extractTimestamp: function(event) {
        return parseFloat(/^\[?([^\]]*)/.exec(event)[1]) * 1000;
      }
    };
  }

  return TimelinePlugin.parseLog(logLines, sourceName, actions, referenceStart);
};


/**
 * Parse a selenium log.
 *
 * @param {Array<Object>} entries The list of entries.
 * @param {string} sourceName Descripton of source.
 * @param {isEventStart: function,
           isEventEnd: function,
           extractCommand: function,
           extractTimestamp: function} actions Methods to interpret entries.
 * @param {number} referenceStart Date in millis.
 */
TimelinePlugin.parseLog =
    function(entries, sourceName, actions, referenceStart) {
  var parsedTimeline = [];
  var currentEvent = {};
  var index = 0;
  var relativeStartTime = 0;
  for (var j = 0; j < entries.length; ++j) {
    var event = entries[j];
    if (actions.isEventStart(event)) {
      currentEvent = {
        source: sourceName,
        id: index++,
        command: actions.extractCommand(event),
        start: actions.extractTimestamp(event)
      };
      if (!relativeStartTime &&
          currentEvent.command.toString() == 'setScriptTimeout' ||
          currentEvent.command.toString() == 'set script timeout' ||
          // [sic], the timeoutt typo is present in the logs
          currentEvent.command.toString() == 'set script timeoutt' ||
          currentEvent.command.toString() == 'SetScriptTimeout') {
        relativeStartTime = currentEvent.start;
      }
    } else if (actions.isEventEnd(event)) {
      currentEvent.end = actions.extractTimestamp(event);
      currentEvent.duration = currentEvent.end - currentEvent.start;
      parsedTimeline.push(currentEvent);
    }
  }

  // Make all the times relative to the first time log types is fetched.
  for (var k = 0; k < parsedTimeline.length; ++k) {
    parsedTimeline[k].start += (referenceStart - relativeStartTime);
    parsedTimeline[k].end += (referenceStart - relativeStartTime);
  }

  return parsedTimeline;
};

TimelinePlugin.prototype.outputResults = function(done) {
  try {
    fs.mkdirSync(this.outdir);
  } catch (e) {
    if (e.code != 'EEXIST') throw e;
  }
  var stream = fs.createReadStream(
      path.join(__dirname, 'indextemplate.html'));
  var outfile = path.join(this.outdir, 'timeline.json');
  fs.writeFileSync(outfile, JSON.stringify(this.timeline));
  stream.pipe(fs.createWriteStream(path.join(this.outdir, 'index.html')));
  stream.on('end', done);
};

/**
 * @param {Object} config The configuration file for the ngHint plugin.
 */
TimelinePlugin.prototype.setup = function(config) {
  var self = this;
  var deferred = q.defer();
  self.outdir = path.resolve(process.cwd(), config.outdir);
  var counter = 0;

  // Override executor so that we get information about commands starting
  // and stopping.
  var originalExecute = browser.driver.executor_.execute;
  browser.driver.executor_.execute = function(command, callback) {
    var timelineEvent = {
      source: 'Test Process',
      id: counter++,
      command: command,
      start: new Date().getTime(),
      end: null
    };
    if (!self.testProcessSetTimeoutTimestamp &&
        timelineEvent.command.name_ == 'setScriptTimeout') {
      self.testProcessSetTimeoutTimestamp = timelineEvent.start;
    }
    self.timeline.push(timelineEvent);
    var wrappedCallback = function(var_args) {
      timelineEvent.end = new Date().getTime();
      callback.apply(this, arguments);
    };
    originalExecute.apply(browser.driver.executor_, [command, wrappedCallback]);
  };

  // Clear the logs here.
  browser.manage().logs().getAvailableLogTypes().then(function(result) {
    // The Selenium standalone server stores its logs in the 'client' channel.
    if (result.indexOf('client') !== -1) {
      self.clientLogAvailable = true;
      deferred.resolve();
      // browser.manage().logs().get('client').then(function() {
      //   deferred.resolve();
      // });
    } else {
      deferred.resolve();
    }
  }, function(error) {
    // No logs are available - this will happen for Internet Explorer, which
    // does not implement webdriver logs. See
    // https://code.google.com/p/selenium/issues/detail?id=4925
    deferred.resolve();
  });
  return deferred.promise;
};

/**
 * @param {Object} config The configuration file for the ngHint plugin.
 */
TimelinePlugin.prototype.teardown = function(config) {
  var self = this;
  var deferred = q.defer();
  // This will be needed later for grabbing data from Sauce Labs.
  browser.getSession().then(function(session) {
    self.sessionId = session.getId();
  });

  // If running with a Selenium Standalone server, get the client logs.
  if (self.clientLogAvailable) {
    browser.manage().logs().get('client').then(function(result) {
      var serverTimeline = TimelinePlugin.parseArrayLog(
          result, 'Selenium Client', self.testProcessSetTimeoutTimestamp);
      self.timeline = self.timeline.concat(serverTimeline);
      deferred.resolve();
    });
  } else {
    deferred.resolve();
  }
  return deferred.promise;
};

/**
 * @param {Object} config The configuration file for the ngHint plugin.
 */
TimelinePlugin.prototype.postResults = function(config) {
  var self = this;
  var deferred = q.defer();
  // We can't get Chrome or IE logs from Sauce Labs via the webdriver logs API
  // because it does not expose them.
  // TODO - if the feature request at
  // https://support.saucelabs.com/entries/60070884-Enable-grabbing-server-logs-from-the-wire-protocol
  // gets implemented, remove this hack.
  if (config.sauceUser && config.sauceKey) {
    // WARNING, HACK: we have a timeout to deal with the fact that there's a
    // delay before Sauce Labs updates logs.
    setTimeout(function() {
      var sauceServer = new SauceLabs({
        username: config.sauceUser,
        password: config.sauceKey
      });

      sauceServer.showJob(self.sessionId, function(err, job) {
        var sauceLog = '';
        if (!job.log_url) {
          console.log('WARNING - no Sauce Labs log url found');
          deferred.resolve();
          return;
        }
        https.get(job.log_url, function(res) {
          res.on('data', function(data) {
            sauceLog += data;
          });

          res.on('end', function() {
            var sauceTimeline =
                TimelinePlugin.parseTextLog(
                    sauceLog,
                    'SauceLabs Server',
                    self.testProcessSetTimeoutTimestamp);
            self.timeline = self.timeline.concat(sauceTimeline);
            self.outputResults(deferred.resolve);
          });

        }).on('error', function(e) {
          console.error(e);
        });
      });
    }, SAUCE_LOGS_WAIT);
  } else {
    self.outputResults(deferred.resolve);
  }
  return deferred.promise;
};


// Export

var timelinePlugin = new TimelinePlugin();

exports.setup = timelinePlugin.setup.bind(timelinePlugin);
exports.teardown = timelinePlugin.teardown.bind(timelinePlugin);
exports.postResults = timelinePlugin.postResults.bind(timelinePlugin);
exports.TimelinePlugin = TimelinePlugin;
