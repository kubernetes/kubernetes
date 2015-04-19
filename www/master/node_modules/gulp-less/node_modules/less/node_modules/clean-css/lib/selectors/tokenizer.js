module.exports = function Tokenizer(data, minifyContext) {
  var chunker = new Chunker(data, 128);
  var chunk = chunker.next();
  var flatBlock = /(^@(font\-face|page|\-ms\-viewport|\-o\-viewport|viewport)|\\@.+?)/;

  var whatsNext = function(context) {
    var cursor = context.cursor;
    var mode = context.mode;
    var closest;

    if (chunk.length == context.cursor) {
      if (chunker.isEmpty())
        return null;

      chunk = chunker.next();
      context.cursor = 0;
    }

    if (mode == 'body') {
      closest = chunk.indexOf('}', cursor);
      return closest > -1 ?
        [closest, 'bodyEnd'] :
        null;
    }

    var nextSpecial = chunk.indexOf('@', context.cursor);
    var nextEscape = mode == 'top' ? chunk.indexOf('__ESCAPED_COMMENT_CLEAN_CSS', context.cursor) : -1;
    var nextBodyStart = chunk.indexOf('{', context.cursor);
    var nextBodyEnd = chunk.indexOf('}', context.cursor);

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
  };

  var tokenize = function(context) {
    var tokenized = [];

    context = context || { cursor: 0, mode: 'top' };

    while (true) {
      var next = whatsNext(context);
      if (!next) {
        var whatsLeft = chunk.substring(context.cursor);
        if (whatsLeft.length > 0) {
          tokenized.push(whatsLeft);
          context.cursor += whatsLeft.length;
        }
        break;
      }

      var nextSpecial = next[0];
      var what = next[1];
      var nextEnd;
      var oldMode;

      if (what == 'special') {
        var firstOpenBraceAt = chunk.indexOf('{', nextSpecial);
        var firstSemicolonAt = chunk.indexOf(';', nextSpecial);
        var isSingle = firstSemicolonAt > -1 && (firstOpenBraceAt == -1 || firstSemicolonAt < firstOpenBraceAt);
        var isBroken = firstOpenBraceAt == -1 && firstSemicolonAt == -1;
        if (isBroken) {
          minifyContext.warnings.push('Broken declaration: \'' + chunk.substring(context.cursor) +  '\'.');
          context.cursor = chunk.length;
        } else if (isSingle) {
          nextEnd = chunk.indexOf(';', nextSpecial + 1);
          tokenized.push(chunk.substring(context.cursor, nextEnd + 1));

          context.cursor = nextEnd + 1;
        } else {
          nextEnd = chunk.indexOf('{', nextSpecial + 1);
          var block = chunk.substring(context.cursor, nextEnd).trim();

          var isFlat = flatBlock.test(block);
          oldMode = context.mode;
          context.cursor = nextEnd + 1;
          context.mode = isFlat ? 'body' : 'block';
          var specialBody = tokenize(context);
          context.mode = oldMode;

          tokenized.push({ block: block, body: specialBody });
        }
      } else if (what == 'escape') {
        nextEnd = chunk.indexOf('__', nextSpecial + 1);
        var escaped = chunk.substring(context.cursor, nextEnd + 2);
        tokenized.push(escaped);

        context.cursor = nextEnd + 2;
      } else if (what == 'bodyStart') {
        var selector = chunk.substring(context.cursor, nextSpecial).trim();

        oldMode = context.mode;
        context.cursor = nextSpecial + 1;
        context.mode = 'body';
        var body = tokenize(context);
        context.mode = oldMode;

        tokenized.push({ selector: selector, body: body });
      } else if (what == 'bodyEnd') {
        // extra closing brace at the top level can be safely ignored
        if (context.mode == 'top') {
          var at = context.cursor;
          var warning = chunk[context.cursor] == '}' ?
            'Unexpected \'}\' in \'' + chunk.substring(at - 20, at + 20) + '\'. Ignoring.' :
            'Unexpected content: \'' + chunk.substring(at, nextSpecial + 1) + '\'. Ignoring.';

          minifyContext.warnings.push(warning);
          context.cursor = nextSpecial + 1;
          continue;
        }

        if (context.mode != 'block')
          tokenized = chunk.substring(context.cursor, nextSpecial);

        context.cursor = nextSpecial + 1;

        break;
      }
    }

    return tokenized;
  };

  return {
    process: function() {
      return tokenize();
    }
  };
};

// Divides `data` into chunks of `chunkSize` for faster processing
var Chunker = function(data, chunkSize) {
  var chunks = [];
  for (var cursor = 0, dataSize = data.length; cursor < dataSize;) {
    var nextCursor = cursor + chunkSize > dataSize ?
      dataSize - 1 :
      cursor + chunkSize;

    if (data[nextCursor] != '}')
      nextCursor = data.indexOf('}', nextCursor);
    if (nextCursor == -1)
      nextCursor = data.length - 1;

    chunks.push(data.substring(cursor, nextCursor + 1));
    cursor = nextCursor + 1;
  }

  return {
    isEmpty: function() {
      return chunks.length === 0;
    },

    next: function() {
      return chunks.shift() || '';
    }
  };
};
