var fs = require('fs');
var path = require('path');
var http = require('http');
var https = require('https');
var url = require('url');

var UrlRewriter = require('../images/url-rewriter');
var Splitter = require('../text/splitter.js');

var merge = function(source1, source2) {
  var target = {};
  for (var key1 in source1)
    target[key1] = source1[key1];
  for (var key2 in source2)
    target[key2] = source2[key2];

  return target;
};

module.exports = function Inliner(context, options) {
  var defaultOptions = {
    timeout: 5000,
    request: {}
  };
  var inlinerOptions = merge(defaultOptions, options || {});

  var process = function(data, options) {
    if (options.shallow) {
      options.shallow = false;
      options._shared.done.push(data);
      return processNext(options);
    }

    options._shared = options._shared || {
      done: [],
      left: []
    };
    var shared = options._shared;

    var nextStart = 0;
    var nextEnd = 0;
    var cursor = 0;
    var isComment = commentScanner(data);
    var afterContent = contentScanner(data);

    options.relativeTo = options.relativeTo || options.root;
    options._baseRelativeTo = options._baseRelativeTo || options.relativeTo;
    options.visited = options.visited || [];

    for (; nextEnd < data.length;) {
      nextStart = nextImportAt(data, cursor);
      if (nextStart == -1)
        break;

      if (isComment(nextStart)) {
        cursor = nextStart + 1;
        continue;
      }

      nextEnd = data.indexOf(';', nextStart);
      if (nextEnd == -1) {
        cursor = data.length;
        data = '';
        break;
      }

      shared.done.push(data.substring(0, nextStart));
      shared.left.unshift([data.substring(nextEnd + 1), options]);

      return afterContent(nextStart) ?
        processNext(options) :
        inline(data, nextStart, nextEnd, options);
    }

    // no @import matched in current data
    shared.done.push(data);
    return processNext(options);
  };

  var nextImportAt = function (data, cursor) {
    var nextLowerCase = data.indexOf('@import', cursor);
    var nextUpperCase = data.indexOf('@IMPORT', cursor);

    if (nextLowerCase > -1 && nextUpperCase == -1)
      return nextLowerCase;
    else if (nextLowerCase == -1 && nextUpperCase > -1)
      return nextUpperCase;
    else
      return Math.min(nextLowerCase, nextUpperCase);
  };

  var processNext = function(options) {
    if (options._shared.left.length > 0)
      return process.apply(null, options._shared.left.shift());
    else
      return options.whenDone(options._shared.done.join(''));
  };

  var commentScanner = function(data) {
    var commentRegex = /(\/\*(?!\*\/)[\s\S]*?\*\/)/;
    var lastStartIndex = 0;
    var lastEndIndex = 0;
    var noComments = false;

    // test whether an index is located within a comment
    var scanner = function(idx) {
      var comment;
      var localStartIndex = 0;
      var localEndIndex = 0;
      var globalStartIndex = 0;
      var globalEndIndex = 0;

      // return if we know there are no more comments
      if (noComments)
        return false;

      // idx can be still within last matched comment (many @import statements inside one comment)
      if (idx > lastStartIndex && idx < lastEndIndex)
        return true;

      comment = data.match(commentRegex);

      if (!comment) {
        noComments = true;
        return false;
      }

      // get the indexes relative to the current data chunk
      lastStartIndex = localStartIndex = comment.index;
      localEndIndex = localStartIndex + comment[0].length;

      // calculate the indexes relative to the full original data
      globalEndIndex = localEndIndex + lastEndIndex;
      globalStartIndex = globalEndIndex - comment[0].length;

      // chop off data up to and including current comment block
      data = data.substring(localEndIndex);
      lastEndIndex = globalEndIndex;

      // re-run scan if comment ended before the idx
      if (globalEndIndex < idx)
        return scanner(idx);

      return globalEndIndex > idx && idx > globalStartIndex;
    };

    return scanner;
  };

  var contentScanner = function(data) {
    var isComment = commentScanner(data);
    var firstContentIdx = -1;
    while (true) {
      firstContentIdx = data.indexOf('{', firstContentIdx + 1);
      if (firstContentIdx == -1 || !isComment(firstContentIdx))
        break;
    }

    return function(idx) {
      return firstContentIdx > -1 ?
        idx > firstContentIdx :
        false;
    };
  };

  var inline = function(data, nextStart, nextEnd, options) {
    options.shallow = data.indexOf('@shallow') > 0;

    var importDeclaration = data
      .substring(nextImportAt(data, nextStart) + '@import'.length + 1, nextEnd)
      .replace(/@shallow\)$/, ')')
      .trim();

    var viaUrl = importDeclaration.indexOf('url(') === 0;
    var urlStartsAt = viaUrl ? 4 : 0;
    var isQuoted = /^['"]/.exec(importDeclaration.substring(urlStartsAt, urlStartsAt + 2));
    var urlEndsAt = isQuoted ?
      importDeclaration.indexOf(isQuoted[0], urlStartsAt + 1) :
      new Splitter(' ').split(importDeclaration)[0].length - (viaUrl ? 1 : 0);

    var importedFile = importDeclaration
      .substring(urlStartsAt, urlEndsAt)
      .replace(/['"]/g, '')
      .replace(/\)$/, '')
      .trim();

    var mediaQuery = importDeclaration
      .substring(urlEndsAt + 1)
      .replace(/^\)/, '')
      .trim();

    var isRemote = options.isRemote ||
      /^(http|https):\/\//.test(importedFile) ||
      /^\/\//.test(importedFile);

    if (options.localOnly && isRemote) {
      context.warnings.push('Ignoring remote @import declaration of "' + importedFile + '" as no callback given.');
      restoreImport(importedFile, mediaQuery, options);

      return processNext(options);
    }

    var method = isRemote ? inlineRemoteResource : inlineLocalResource;
    return method(importedFile, mediaQuery, options);
  };

  var inlineRemoteResource = function(importedFile, mediaQuery, options) {
    var importedUrl = /^https?:\/\//.test(importedFile) ?
      importedFile :
      url.resolve(options.relativeTo, importedFile);

    if (importedUrl.indexOf('//') === 0)
      importedUrl = 'http:' + importedUrl;

    if (options.visited.indexOf(importedUrl) > -1)
      return processNext(options);


    if (context.debug)
      console.error('Inlining remote stylesheet: ' + importedUrl);

    options.visited.push(importedUrl);

    var get = importedUrl.indexOf('http://') === 0 ?
      http.get :
      https.get;

    var timedOut = false;
    var handleError = function(message) {
      context.errors.push('Broken @import declaration of "' + importedUrl + '" - ' + message);
      restoreImport(importedUrl, mediaQuery, options);

      processNext(options);
    };
    var requestOptions = merge(url.parse(importedUrl), inlinerOptions.request);

    get(requestOptions, function(res) {
      if (res.statusCode < 200 || res.statusCode > 399) {
        return handleError('error ' + res.statusCode);
      } else if (res.statusCode > 299) {
        var movedUrl = url.resolve(importedUrl, res.headers.location);
        return inlineRemoteResource(movedUrl, mediaQuery, options);
      }

      var chunks = [];
      var parsedUrl = url.parse(importedUrl);
      res.on('data', function(chunk) {
        chunks.push(chunk.toString());
      });
      res.on('end', function() {
        var importedData = chunks.join('');
        importedData = UrlRewriter.process(importedData, { toBase: importedUrl });

        if (mediaQuery.length > 0)
          importedData = '@media ' + mediaQuery + '{' + importedData + '}';

        process(importedData, {
          isRemote: true,
          relativeTo: parsedUrl.protocol + '//' + parsedUrl.host,
          _shared: options._shared,
          whenDone: options.whenDone,
          visited: options.visited,
          shallow: options.shallow
        });
      });
    })
    .on('error', function(res) {
      handleError(res.message);
    })
    .on('timeout', function() {
      // FIX: node 0.8 fires this event twice
      if (timedOut)
        return;

      handleError('timeout');
      timedOut = true;
    })
    .setTimeout(inlinerOptions.timeout);
  };

  var inlineLocalResource = function(importedFile, mediaQuery, options) {
    var relativeTo = importedFile[0] == '/' ?
      options.root :
      options.relativeTo;

    var fullPath = path.resolve(path.join(relativeTo, importedFile));

    if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
      context.errors.push('Broken @import declaration of "' + importedFile + '"');
      return processNext(options);
    }

    if (options.visited.indexOf(fullPath) > -1)
      return processNext(options);


    if (context.debug)
      console.error('Inlining local stylesheet: ' + fullPath);

    options.visited.push(fullPath);

    var importedData = fs.readFileSync(fullPath, 'utf8');
    var importRelativeTo = path.dirname(fullPath);
    importedData = UrlRewriter.process(importedData, {
      relative: true,
      fromBase: importRelativeTo,
      toBase: options._baseRelativeTo
    });

    if (mediaQuery.length > 0)
      importedData = '@media ' + mediaQuery + '{' + importedData + '}';

    return process(importedData, {
      root: options.root,
      relativeTo: importRelativeTo,
      _baseRelativeTo: options._baseRelativeTo,
      _shared: options._shared,
      visited: options.visited,
      whenDone: options.whenDone,
      localOnly: options.localOnly,
      shallow: options.shallow
    });
  };

  var restoreImport = function(importedUrl, mediaQuery, options) {
    var restoredImport = '@import url(' + importedUrl + ')' + (mediaQuery.length > 0 ? ' ' + mediaQuery : '') + ';';
    options._shared.done.push(restoredImport);
  };

  // Inlines all imports taking care of repetitions, unknown files, and circular dependencies
  return { process: process };
};
