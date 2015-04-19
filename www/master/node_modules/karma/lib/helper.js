var fs   = require('fs');
var path = require('path');
var _    = require('lodash');
var useragent = require('useragent');


exports.browserFullNameToShort = function(fullName) {
  var agent = useragent.parse(fullName);
  return agent.toAgent() + ' (' + agent.os + ')';
};


exports.isDefined = function(value) {
  return !_.isUndefined(value);
};

exports.isFunction = _.isFunction;
exports.isString = _.isString;
exports.isObject = _.isObject;
exports.isArray = _.isArray;

var ABS_URL = /^https?:\/\//;
exports.isUrlAbsolute = function(url) {
  return ABS_URL.test(url);
};


exports.camelToSnake = function(camelCase) {
  return camelCase.replace(/[A-Z]/g, function(match, pos) {
    return (pos > 0 ? '_' : '') + match.toLowerCase();
  });
};


exports.ucFirst = function(word) {
  return word.charAt(0).toUpperCase() + word.substr(1);
};


exports.dashToCamel = function(dash) {
  var words = dash.split('-');
  return words.shift() + words.map(exports.ucFirst).join('');
};


exports.arrayRemove = function(collection, item) {
  var idx = collection.indexOf(item);

  if (idx !== -1) {
    collection.splice(idx, 1);
    return true;
  }

  return false;
};


exports.merge = function() {
  var args = Array.prototype.slice.call(arguments, 0);
  args.unshift({});
  return _.merge.apply({}, args);
};


exports.formatTimeInterval = function(time) {
  var mins = Math.floor(time / 60000);
  var secs = (time - mins * 60000) / 1000;
  var str = secs + (secs === 1 ? ' sec' : ' secs');

  if (mins) {
    str = mins + (mins === 1 ? ' min ' : ' mins ') + str;
  }

  return str;
};

var replaceWinPath = function(path) {
  return exports.isDefined(path) ? path.replace(/\\/g, '/') : path;
};

exports.normalizeWinPath = process.platform === 'win32' ? replaceWinPath : _.identity;

exports.mkdirIfNotExists = function mkdir(directory, done) {
  // TODO(vojta): handle if it's a file
  fs.stat(directory, function(err, stat) {
    if (stat && stat.isDirectory()) {
      done();
    } else {
      mkdir(path.dirname(directory), function() {
        fs.mkdir(directory, done);
      });
    }
  });
};

// export lodash
exports._ = _;
