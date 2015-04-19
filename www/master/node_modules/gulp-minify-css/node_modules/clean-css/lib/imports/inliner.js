var fs = require('fs');
var path = require('path');
var http = require('http');
var https = require('https');
var url = require('url');

var UrlRewriter = require('../images/url-rewriter');
var Splitter = require('../utils/splitter.js');
var override = require('../utils/object.js').override;

var MAP_MARKER = /\/\*# sourceMappingURL=(\S+) \*\//;
var REMOTE_RESOURCE = /^(https?:)?\/\//;

function ImportInliner (context) {
  this.outerContext = context;
}

ImportInliner.prototype.process = function (data, context) {
  var root = this.outerContext.options.root || process.cwd();

  context = override(context, {
    baseRelativeTo: this.outerContext.options.relativeTo || root,
    debug: this.outerContext.options.debug,
    done: [],
    errors: this.outerContext.errors,
    left: [],
    inliner: this.outerContext.options.inliner,
    rebase: this.outerContext.options.rebase,
    relativeTo: this.outerContext.options.relativeTo || root,
    root: root,
    sourceTracker: this.outerContext.sourceTracker,
    warnings: this.outerContext.warnings,
    visited: []
  });

  return importFrom(data, context);
};

function importFrom(data, context) {
  if (context.shallow) {
    context.shallow = false;
    context.done.push(data);
    return processNext(context);
  }

  var nextStart = 0;
  var nextEnd = 0;
  var cursor = 0;
  var isComment = commentScanner(data);
  var afterContent = contentScanner(data);

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

    context.done.push(data.substring(0, nextStart));
    context.left.unshift([data.substring(nextEnd + 1), context]);

    return afterContent(nextStart) ?
      processNext(context) :
      inline(data, nextStart, nextEnd, context);
  }

  // no @import matched in current data
  context.done.push(data);
  return processNext(context);
}

function rebaseMap(data, source) {
  return data.replace(MAP_MARKER, function (match, sourceMapUrl) {
    return REMOTE_RESOURCE.test(sourceMapUrl) ?
      match :
      match.replace(sourceMapUrl, url.resolve(source, sourceMapUrl));
  });
}

function nextImportAt(data, cursor) {
  var nextLowerCase = data.indexOf('@import', cursor);
  var nextUpperCase = data.indexOf('@IMPORT', cursor);

  if (nextLowerCase > -1 && nextUpperCase == -1)
    return nextLowerCase;
  else if (nextLowerCase == -1 && nextUpperCase > -1)
    return nextUpperCase;
  else
    return Math.min(nextLowerCase, nextUpperCase);
}

function processNext(context) {
  return context.left.length > 0 ?
    importFrom.apply(null, context.left.shift()) :
    context.whenDone(context.done.join(''));
}

function commentScanner(data) {
  var commentRegex = /(\/\*(?!\*\/)[\s\S]*?\*\/)/;
  var lastStartIndex = 0;
  var lastEndIndex = 0;
  var noComments = false;

  // test whether an index is located within a comment
  return function scanner(idx) {
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
}

function contentScanner(data) {
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
}

function inline(data, nextStart, nextEnd, context) {
  context.shallow = data.indexOf('@shallow') > 0;

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

  var isRemote = context.isRemote || REMOTE_RESOURCE.test(importedFile);

  if (context.localOnly && isRemote) {
    context.warnings.push('Ignoring remote @import declaration of "' + importedFile + '" as no callback given.');
    restoreImport(importedFile, mediaQuery, context);

    return processNext(context);
  }

  var method = isRemote ? inlineRemoteResource : inlineLocalResource;
  return method(importedFile, mediaQuery, context);
}

function inlineRemoteResource(importedFile, mediaQuery, context) {
  var importedUrl = REMOTE_RESOURCE.test(importedFile) ?
    importedFile :
    url.resolve(context.relativeTo, importedFile);

  if (importedUrl.indexOf('//') === 0)
    importedUrl = 'http:' + importedUrl;

  if (context.visited.indexOf(importedUrl) > -1)
    return processNext(context);


  if (context.debug)
    console.error('Inlining remote stylesheet: ' + importedUrl);

  context.visited.push(importedUrl);

  var get = importedUrl.indexOf('http://') === 0 ?
    http.get :
    https.get;

  function handleError(message) {
    context.errors.push('Broken @import declaration of "' + importedUrl + '" - ' + message);
    restoreImport(importedUrl, mediaQuery, context);

    processNext(context);
  }

  var requestOptions = override(url.parse(importedUrl), context.inliner.request);
  get(requestOptions, function(res) {
    if (res.statusCode < 200 || res.statusCode > 399) {
      return handleError('error ' + res.statusCode);
    } else if (res.statusCode > 299) {
      var movedUrl = url.resolve(importedUrl, res.headers.location);
      return inlineRemoteResource(movedUrl, mediaQuery, context);
    }

    var chunks = [];
    var parsedUrl = url.parse(importedUrl);
    res.on('data', function(chunk) {
      chunks.push(chunk.toString());
    });
    res.on('end', function() {
      var importedData = chunks.join('');
      if (context.rebase)
        importedData = new UrlRewriter({ toBase: importedUrl }).process(importedData);
      importedData = context.sourceTracker.store(importedUrl, importedData);
      importedData = rebaseMap(importedData, importedUrl);

      if (mediaQuery.length > 0)
        importedData = '@media ' + mediaQuery + '{' + importedData + '}';

      var newContext = override(context, {
        isRemote: true,
        relativeTo: parsedUrl.protocol + '//' + parsedUrl.host
      });

      importFrom(importedData, newContext);
    });
  })
  .on('error', function(res) {
    handleError(res.message);
  })
  .on('timeout', function() {
    handleError('timeout');
  })
  .setTimeout(context.inliner.timeout);
}

function inlineLocalResource(importedFile, mediaQuery, context) {
  var relativeTo = importedFile[0] == '/' ?
    context.root :
    context.relativeTo;

  var fullPath = path.resolve(path.join(relativeTo, importedFile));

  if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
    context.errors.push('Broken @import declaration of "' + importedFile + '"');
    return processNext(context);
  }

  if (context.visited.indexOf(fullPath) > -1)
    return processNext(context);


  if (context.debug)
    console.error('Inlining local stylesheet: ' + fullPath);

  context.visited.push(fullPath);

  var importRelativeTo = path.dirname(fullPath);
  var importedData = fs.readFileSync(fullPath, 'utf8');
  if (context.rebase) {
    var rewriter = new UrlRewriter({
      relative: true,
      fromBase: importRelativeTo,
      toBase: context.baseRelativeTo
    });
    importedData = rewriter.process(importedData);
  }
  importedData = context.sourceTracker.store(path.resolve(context.relativeTo, fullPath), importedData);

  if (mediaQuery.length > 0)
    importedData = '@media ' + mediaQuery + '{' + importedData + '}';

  var newContext = override(context, {
    relativeTo: importRelativeTo,
  });

  return importFrom(importedData, newContext);
}

function restoreImport(importedUrl, mediaQuery, context) {
  var restoredImport = '@import url(' + importedUrl + ')' + (mediaQuery.length > 0 ? ' ' + mediaQuery : '') + ';';
  context.done.push(restoredImport);
}

module.exports = ImportInliner;
