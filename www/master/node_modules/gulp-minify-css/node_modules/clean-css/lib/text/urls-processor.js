var EscapeStore = require('./escape-store');

var URL_PREFIX = 'url(';
var URL_SUFFIX = ')';
var lineBreak = require('os').EOL;

function UrlsProcessor(context, saveWaypoints) {
  this.urls = new EscapeStore('URL');
  this.context = context;
  this.saveWaypoints = saveWaypoints;
}

// Strip urls by replacing them by a special
// marker for further restoring. It's done via string scanning
// instead of regexps to speed up the process.
UrlsProcessor.prototype.escape = function (data) {
  var nextStart = 0;
  var nextEnd = 0;
  var cursor = 0;
  var tempData = [];
  var breaksCount;
  var lastBreakAt;
  var indent;
  var saveWaypoints = this.saveWaypoints;

  for (; nextEnd < data.length;) {
    nextStart = data.indexOf(URL_PREFIX, nextEnd);
    if (nextStart == -1)
      break;

    if (data[nextStart + URL_PREFIX.length] == '"')
      nextEnd = data.indexOf('"', nextStart + URL_PREFIX.length + 1);
    else if (data[nextStart + URL_PREFIX.length] == '\'')
      nextEnd = data.indexOf('\'', nextStart + URL_PREFIX.length + 1);
    else
      nextEnd = data.indexOf(URL_SUFFIX, nextStart);

    // Following lines are a safety mechanism to ensure
    // incorrectly terminated urls are processed correctly.
    if (nextEnd == -1) {
      nextEnd = data.indexOf('}', nextStart);

      if (nextEnd == -1)
        nextEnd = data.length;
      else
        nextEnd--;

      this.context.warnings.push('Broken URL declaration: \'' + data.substring(nextStart, nextEnd + 1) + '\'.');
    } else {
      if (data[nextEnd] != URL_SUFFIX)
        nextEnd = data.indexOf(URL_SUFFIX, nextEnd);
    }

    var url = data.substring(nextStart, nextEnd + 1);

    if (saveWaypoints) {
      breaksCount = url.split(lineBreak).length - 1;
      lastBreakAt = url.lastIndexOf(lineBreak);
      indent = lastBreakAt > 0 ?
        url.substring(lastBreakAt + lineBreak.length).length :
        url.length;
    }

    var placeholder = this.urls.store(url, saveWaypoints ? [breaksCount, indent] : null);
    tempData.push(data.substring(cursor, nextStart));
    tempData.push(placeholder);

    cursor = nextEnd + 1;
  }

  return tempData.length > 0 ?
    tempData.join('') + data.substring(cursor, data.length) :
    data;
};

function normalize(url) {
  url = url
    .replace(/\\?\n|\\?\r\n/g, '')
    .replace(/(\s{2,}|\s)/g, ' ')
    .replace(/^url\((['"])? /, 'url($1')
    .replace(/ (['"])?\)$/, '$1)');

  if (!/url\(.*[\s\(\)].*\)/.test(url) && !/url\(['"]data:[^;]+;charset/.test(url))
    url = url.replace(/["']/g, '');

  return url;
}

UrlsProcessor.prototype.restore = function (data) {
  var tempData = [];
  var cursor = 0;

  for (; cursor < data.length;) {
    var nextMatch = this.urls.nextMatch(data, cursor);
    if (nextMatch.start < 0)
      break;

    tempData.push(data.substring(cursor, nextMatch.start));
    var url = normalize(this.urls.restore(nextMatch.match));
    tempData.push(url);

    cursor = nextMatch.end;
  }

  return tempData.length > 0 ?
    tempData.join('') + data.substring(cursor, data.length) :
    data;
};

module.exports = UrlsProcessor;
