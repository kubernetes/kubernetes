define(
  ["../exception","exports"],
  function(__dependency1__, __exports__) {
    "use strict";
    var Exception = __dependency1__["default"];

    function stripFlags(open, close) {
      return {
        left: open.charAt(2) === '~',
        right: close.charAt(close.length-3) === '~'
      };
    }

    __exports__.stripFlags = stripFlags;
    function prepareBlock(mustache, program, inverseAndProgram, close, inverted, locInfo) {
      /*jshint -W040 */
      if (mustache.sexpr.id.original !== close.path.original) {
        throw new Exception(mustache.sexpr.id.original + ' doesn\'t match ' + close.path.original, mustache);
      }

      var inverse = inverseAndProgram && inverseAndProgram.program;

      var strip = {
        left: mustache.strip.left,
        right: close.strip.right,

        // Determine the standalone candiacy. Basically flag our content as being possibly standalone
        // so our parent can determine if we actually are standalone
        openStandalone: isNextWhitespace(program.statements),
        closeStandalone: isPrevWhitespace((inverse || program).statements)
      };

      if (mustache.strip.right) {
        omitRight(program.statements, null, true);
      }

      if (inverse) {
        var inverseStrip = inverseAndProgram.strip;

        if (inverseStrip.left) {
          omitLeft(program.statements, null, true);
        }
        if (inverseStrip.right) {
          omitRight(inverse.statements, null, true);
        }
        if (close.strip.left) {
          omitLeft(inverse.statements, null, true);
        }

        // Find standalone else statments
        if (isPrevWhitespace(program.statements)
            && isNextWhitespace(inverse.statements)) {

          omitLeft(program.statements);
          omitRight(inverse.statements);
        }
      } else {
        if (close.strip.left) {
          omitLeft(program.statements, null, true);
        }
      }

      if (inverted) {
        return new this.BlockNode(mustache, inverse, program, strip, locInfo);
      } else {
        return new this.BlockNode(mustache, program, inverse, strip, locInfo);
      }
    }

    __exports__.prepareBlock = prepareBlock;
    function prepareProgram(statements, isRoot) {
      for (var i = 0, l = statements.length; i < l; i++) {
        var current = statements[i],
            strip = current.strip;

        if (!strip) {
          continue;
        }

        var _isPrevWhitespace = isPrevWhitespace(statements, i, isRoot, current.type === 'partial'),
            _isNextWhitespace = isNextWhitespace(statements, i, isRoot),

            openStandalone = strip.openStandalone && _isPrevWhitespace,
            closeStandalone = strip.closeStandalone && _isNextWhitespace,
            inlineStandalone = strip.inlineStandalone && _isPrevWhitespace && _isNextWhitespace;

        if (strip.right) {
          omitRight(statements, i, true);
        }
        if (strip.left) {
          omitLeft(statements, i, true);
        }

        if (inlineStandalone) {
          omitRight(statements, i);

          if (omitLeft(statements, i)) {
            // If we are on a standalone node, save the indent info for partials
            if (current.type === 'partial') {
              current.indent = (/([ \t]+$)/).exec(statements[i-1].original) ? RegExp.$1 : '';
            }
          }
        }
        if (openStandalone) {
          omitRight((current.program || current.inverse).statements);

          // Strip out the previous content node if it's whitespace only
          omitLeft(statements, i);
        }
        if (closeStandalone) {
          // Always strip the next node
          omitRight(statements, i);

          omitLeft((current.inverse || current.program).statements);
        }
      }

      return statements;
    }

    __exports__.prepareProgram = prepareProgram;function isPrevWhitespace(statements, i, isRoot) {
      if (i === undefined) {
        i = statements.length;
      }

      // Nodes that end with newlines are considered whitespace (but are special
      // cased for strip operations)
      var prev = statements[i-1],
          sibling = statements[i-2];
      if (!prev) {
        return isRoot;
      }

      if (prev.type === 'content') {
        return (sibling || !isRoot ? (/\r?\n\s*?$/) : (/(^|\r?\n)\s*?$/)).test(prev.original);
      }
    }
    function isNextWhitespace(statements, i, isRoot) {
      if (i === undefined) {
        i = -1;
      }

      var next = statements[i+1],
          sibling = statements[i+2];
      if (!next) {
        return isRoot;
      }

      if (next.type === 'content') {
        return (sibling || !isRoot ? (/^\s*?\r?\n/) : (/^\s*?(\r?\n|$)/)).test(next.original);
      }
    }

    // Marks the node to the right of the position as omitted.
    // I.e. {{foo}}' ' will mark the ' ' node as omitted.
    //
    // If i is undefined, then the first child will be marked as such.
    //
    // If mulitple is truthy then all whitespace will be stripped out until non-whitespace
    // content is met.
    function omitRight(statements, i, multiple) {
      var current = statements[i == null ? 0 : i + 1];
      if (!current || current.type !== 'content' || (!multiple && current.rightStripped)) {
        return;
      }

      var original = current.string;
      current.string = current.string.replace(multiple ? (/^\s+/) : (/^[ \t]*\r?\n?/), '');
      current.rightStripped = current.string !== original;
    }

    // Marks the node to the left of the position as omitted.
    // I.e. ' '{{foo}} will mark the ' ' node as omitted.
    //
    // If i is undefined then the last child will be marked as such.
    //
    // If mulitple is truthy then all whitespace will be stripped out until non-whitespace
    // content is met.
    function omitLeft(statements, i, multiple) {
      var current = statements[i == null ? statements.length - 1 : i - 1];
      if (!current || current.type !== 'content' || (!multiple && current.leftStripped)) {
        return;
      }

      // We omit the last node if it's whitespace only and not preceeded by a non-content node.
      var original = current.string;
      current.string = current.string.replace(multiple ? (/\s+$/) : (/[ \t]+$/), '');
      current.leftStripped = current.string !== original;
      return current.leftStripped;
    }
  });