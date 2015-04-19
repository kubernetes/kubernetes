var Chunker = require('../utils/chunker');
var Extract = require('../utils/extractors');
var SourceMaps = require('../utils/source-maps');

var flatBlock = /(^@(font\-face|page|\-ms\-viewport|\-o\-viewport|viewport|counter\-style)|\\@.+?)/;

function Tokenizer(minifyContext, addMetadata, addSourceMap) {
  this.minifyContext = minifyContext;
  this.addMetadata = addMetadata;
  this.addSourceMap = addSourceMap;
}

Tokenizer.prototype.toTokens = function (data) {
  data = data.replace(/\r\n/g, '\n');

  var chunker = new Chunker(data, '}', 128);
  if (chunker.isEmpty())
    return [];

  var context = {
    cursor: 0,
    mode: 'top',
    chunker: chunker,
    chunk: chunker.next(),
    outer: this.minifyContext,
    addMetadata: this.addMetadata,
    addSourceMap: this.addSourceMap,
    state: [],
    line: 1,
    column: 0,
    source: undefined
  };

  return tokenize(context);
};

function whatsNext(context) {
  var mode = context.mode;
  var chunk = context.chunk;
  var closest;

  if (chunk.length == context.cursor) {
    if (context.chunker.isEmpty())
      return null;

    context.chunk = chunk = context.chunker.next();
    context.cursor = 0;
  }

  if (mode == 'body') {
    closest = chunk.indexOf('}', context.cursor);
    return closest > -1 ?
      [closest, 'bodyEnd'] :
      null;
  }

  var nextSpecial = chunk.indexOf('@', context.cursor);
  var nextEscape = chunk.indexOf('__ESCAPED_', context.cursor);
  var nextBodyStart = chunk.indexOf('{', context.cursor);
  var nextBodyEnd = chunk.indexOf('}', context.cursor);

  if (nextEscape > -1 && /\S/.test(chunk.substring(context.cursor, nextEscape)))
    nextEscape = -1;

  closest = nextSpecial;
  if (closest == -1 || (nextEscape > -1 && nextEscape < closest))
    closest = nextEscape;
  if (closest == -1 || (nextBodyStart > -1 && nextBodyStart < closest))
    closest = nextBodyStart;
  if (closest == -1 || (nextBodyEnd > -1 && nextBodyEnd < closest))
    closest = nextBodyEnd;

  if (closest == -1)
    return;
  if (nextEscape === closest)
    return [closest, 'escape'];
  if (nextBodyStart === closest)
    return [closest, 'bodyStart'];
  if (nextBodyEnd === closest)
    return [closest, 'bodyEnd'];
  if (nextSpecial === closest)
    return [closest, 'special'];
}

function tokenize(context) {
  var chunk = context.chunk;
  var tokenized = [];
  var newToken;
  var value;
  var addSourceMap = context.addSourceMap;

  while (true) {
    var next = whatsNext(context);
    if (!next) {
      var whatsLeft = context.chunk.substring(context.cursor);
      if (whatsLeft.trim().length > 0) {
        tokenized.push({ kind: 'text', value: whatsLeft });
        context.cursor += whatsLeft.length;
      }
      break;
    }

    var nextSpecial = next[0];
    var what = next[1];
    var nextEnd;
    var oldMode;

    chunk = context.chunk;

    if (context.cursor != nextSpecial && what != 'bodyEnd') {
      var spacing = chunk.substring(context.cursor, nextSpecial);
      var leadingWhitespace = /^\s+/.exec(spacing);

      if (leadingWhitespace) {
        context.cursor += leadingWhitespace[0].length;

        if (addSourceMap)
          SourceMaps.track(leadingWhitespace[0], context);
      }
    }

    if (what == 'special') {
      var firstOpenBraceAt = chunk.indexOf('{', nextSpecial);
      var firstSemicolonAt = chunk.indexOf(';', nextSpecial);
      var isSingle = firstSemicolonAt > -1 && (firstOpenBraceAt == -1 || firstSemicolonAt < firstOpenBraceAt);
      var isBroken = firstOpenBraceAt == -1 && firstSemicolonAt == -1;
      if (isBroken) {
        context.outer.warnings.push('Broken declaration: \'' + chunk.substring(context.cursor) +  '\'.');
        context.cursor = chunk.length;
      } else if (isSingle) {
        nextEnd = chunk.indexOf(';', nextSpecial + 1);

        value = chunk.substring(context.cursor, nextEnd + 1);
        newToken = { kind: 'at-rule', value: value };
        tokenized.push(newToken);

        if (addSourceMap)
          newToken.metadata = SourceMaps.saveAndTrack(value, context, true);

        context.cursor = nextEnd + 1;
      } else {
        nextEnd = chunk.indexOf('{', nextSpecial + 1);
        value = chunk.substring(context.cursor, nextEnd);

        var trimmedValue = value.trim();
        var isFlat = flatBlock.test(trimmedValue);
        oldMode = context.mode;
        context.cursor = nextEnd + 1;
        context.mode = isFlat ? 'body' : 'block';

        newToken = { kind: 'block', value: trimmedValue, isFlatBlock: isFlat };

        if (addSourceMap)
          newToken.metadata = SourceMaps.saveAndTrack(value, context, true);

        newToken.body = tokenize(context);
        if (typeof newToken.body == 'string')
          newToken.body = Extract.properties(newToken.body, context).tokenized;

        context.mode = oldMode;

        if (addSourceMap)
          SourceMaps.suffix(context);

        tokenized.push(newToken);
      }
    } else if (what == 'escape') {
      nextEnd = chunk.indexOf('__', nextSpecial + 1);
      var escaped = chunk.substring(context.cursor, nextEnd + 2);
      var isStartSourceMarker = !!context.outer.sourceTracker.nextStart(escaped);
      var isEndSourceMarker = !!context.outer.sourceTracker.nextEnd(escaped);

      if (isStartSourceMarker) {
        if (addSourceMap)
          SourceMaps.track(escaped, context);

        context.state.push({
          source: context.source,
          line: context.line,
          column: context.column
        });
        context.source = context.outer.sourceTracker.nextStart(escaped).filename;
        context.line = 1;
        context.column = 0;
      } else if (isEndSourceMarker) {
        var oldState = context.state.pop();
        context.source = oldState.source;
        context.line = oldState.line;
        context.column = oldState.column;

        if (addSourceMap)
          SourceMaps.track(escaped, context);
      } else {
        if (escaped.indexOf('__ESCAPED_COMMENT_SPECIAL') === 0)
          tokenized.push({ kind: 'text', value: escaped });

        if (addSourceMap)
          SourceMaps.track(escaped, context);
      }

      context.cursor = nextEnd + 2;
    } else if (what == 'bodyStart') {
      var selectorData = Extract.selectors(chunk.substring(context.cursor, nextSpecial), context);

      oldMode = context.mode;
      context.cursor = nextSpecial + 1;
      context.mode = 'body';

      var bodyData = Extract.properties(tokenize(context), context);

      if (addSourceMap)
        SourceMaps.suffix(context);

      context.mode = oldMode;

      newToken = {
        kind: 'selector',
        value: selectorData.tokenized,
        body: bodyData.tokenized
      };
      if (context.addMetadata) {
        newToken.metadata = {
          body: bodyData.list.join(','),
          bodiesList: bodyData.list,
          selector: selectorData.list.join(','),
          selectorsList: selectorData.list
        };
      }
      tokenized.push(newToken);
    } else if (what == 'bodyEnd') {
      // extra closing brace at the top level can be safely ignored
      if (context.mode == 'top') {
        var at = context.cursor;
        var warning = chunk[context.cursor] == '}' ?
          'Unexpected \'}\' in \'' + chunk.substring(at - 20, at + 20) + '\'. Ignoring.' :
          'Unexpected content: \'' + chunk.substring(at, nextSpecial + 1) + '\'. Ignoring.';

        context.outer.warnings.push(warning);
        context.cursor = nextSpecial + 1;
        continue;
      }

      if (context.mode == 'block' && context.addSourceMap)
        SourceMaps.track(chunk.substring(context.cursor, nextSpecial), context);
      if (context.mode != 'block')
        tokenized = chunk.substring(context.cursor, nextSpecial);

      context.cursor = nextSpecial + 1;

      break;
    }
  }

  return tokenized;
}

module.exports = Tokenizer;
