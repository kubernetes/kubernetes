var SourceMapConsumer = require('source-map').SourceMapConsumer;

var fs = require('fs');
var path = require('path');
var http = require('http');
var https = require('https');
var url = require('url');

var override = require('../utils/object.js').override;

var MAP_MARKER = /\/\*# sourceMappingURL=(\S+) \*\//;

function InputSourceMapStore(outerContext) {
  this.options = outerContext.options;
  this.errors = outerContext.errors;
  this.sourceTracker = outerContext.sourceTracker;
  this.timeout = this.options.inliner.timeout;
  this.requestOptions = this.options.inliner.request;

  this.maps = {};
}

function fromString(self, data, whenDone) {
  self.maps[undefined] = new SourceMapConsumer(self.options.sourceMap);
  return whenDone();
}

function fromSource(self, data, whenDone, context) {
  var nextAt = 0;

  function proceedToNext() {
    context.cursor += nextAt + 1;
    fromSource(self, data, whenDone, context);
  }

  while (context.cursor < data.length) {
    var fragment = data.substring(context.cursor);

    var markerStartMatch = self.sourceTracker.nextStart(fragment) || { index: -1 };
    var markerEndMatch = self.sourceTracker.nextEnd(fragment) || { index: -1 };
    var mapMatch = MAP_MARKER.exec(fragment) || { index: -1 };
    var sourceMapFile = mapMatch[1];

    nextAt = data.length;
    if (markerStartMatch.index > -1)
      nextAt = markerStartMatch.index;
    if (markerEndMatch.index > -1 && markerEndMatch.index < nextAt)
      nextAt = markerEndMatch.index;
    if (mapMatch.index > -1 && mapMatch.index < nextAt)
      nextAt = mapMatch.index;

    if (nextAt == data.length)
      break;

    if (nextAt == markerStartMatch.index) {
      context.files.push(markerStartMatch.filename);
    } else if (nextAt == markerEndMatch.index) {
      context.files.pop();
    } else if (nextAt == mapMatch.index) {
      var isRemote = /^https?:\/\//.test(sourceMapFile) || /^\/\//.test(sourceMapFile);
      if (isRemote) {
        return fetchMapFile(self, sourceMapFile, context, proceedToNext);
      } else {
        var sourceFile = context.files[context.files.length - 1];
        var sourceDir = sourceFile ? path.dirname(sourceFile) : self.options.relativeTo;

        var inputMapData = fs.readFileSync(path.join(sourceDir || '', sourceMapFile), 'utf-8');
        self.maps[sourceFile || undefined] = new SourceMapConsumer(inputMapData);
      }
    }

    context.cursor += nextAt + 1;
  }

  return whenDone();
}

function fetchMapFile(self, mapSource, context, done) {
  function handleError(status) {
    context.errors.push('Broken source map at "' + mapSource + '" - ' + status);
    return done();
  }

  var method = mapSource.indexOf('https') === 0 ? https : http;
  var requestOptions = override(url.parse(mapSource), self.requestOptions);

  method
    .get(requestOptions, function (res) {
      if (res.statusCode < 200 || res.statusCode > 299)
        return handleError(res.statusCode);

      var chunks = [];
      res.on('data', function (chunk) {
        chunks.push(chunk.toString());
      });
      res.on('end', function () {
        self.maps[context.files[context.files.length - 1] || undefined] = new SourceMapConsumer(chunks.join(''));
        done();
      });
    })
    .on('error', function(res) {
      handleError(res.message);
    })
    .on('timeout', function() {
      handleError('timeout');
    })
    .setTimeout(self.timeout);
}

function originalPositionIn(trackedSource, sourceInfo, token, allowNFallbacks) {
  // FIXME: we should rather track original positions in tokenizer
  // here it is a bit too late do do it reliably hance the hack
  var originalPosition;
  var maxRange = token.replace(/[>\+~]/g, ' $1 ').length;
  var position = {
    line: sourceInfo.line,
    column: sourceInfo.column + maxRange
  };

  while (maxRange-- > 0) {
    position.column--;
    originalPosition = trackedSource.originalPositionFor(position);

    if (originalPosition)
      break;
  }

  if (originalPosition.line === null && sourceInfo.line > 1 && allowNFallbacks > 0)
    return originalPositionIn(trackedSource, { line: sourceInfo.line - 1, column: sourceInfo.column }, token, allowNFallbacks - 1);

  return originalPosition;
}

InputSourceMapStore.prototype.track = function (data, whenDone) {
  return typeof this.options.sourceMap == 'string' ?
    fromString(this, data, whenDone) :
    fromSource(this, data, whenDone, { files: [], cursor: 0, errors: this.errors });
};

InputSourceMapStore.prototype.isTracking = function (sourceInfo) {
  return !!this.maps[sourceInfo.source];
};

InputSourceMapStore.prototype.originalPositionFor = function (sourceInfo, token, allowNFallbacks) {
  return originalPositionIn(this.maps[sourceInfo.source], sourceInfo, token, allowNFallbacks);
};

module.exports = InputSourceMapStore;
