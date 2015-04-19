var Tokenizer = require('./tokenizer');
var PropertyOptimizer = require('../properties/optimizer');

module.exports = function Optimizer(data, context, options) {
  var specialSelectors = {
    '*': /(\-moz\-|\-ms\-|\-o\-|\-webkit\-|:dir\([a-z-]*\)|:first(?![a-z-])|:fullscreen|:left|:read-only|:read-write|:right)/,
    'ie8': /(\-moz\-|\-ms\-|\-o\-|\-webkit\-|:root|:nth|:first\-of|:last|:only|:empty|:target|:checked|::selection|:enabled|:disabled|:not)/,
    'ie7': /(\-moz\-|\-ms\-|\-o\-|\-webkit\-|:focus|:before|:after|:root|:nth|:first\-of|:last|:only|:empty|:target|:checked|::selection|:enabled|:disabled|:not)/
  };

  var minificationsMade = [];

  var propertyOptimizer = new PropertyOptimizer(options.compatibility, options.aggressiveMerging, context);

  var cleanUpSelector = function(selectors) {
    if (selectors.indexOf(',') == -1)
      return selectors;

    var plain = [];
    var cursor = 0;
    var lastComma = 0;
    var noBrackets = selectors.indexOf('(') == -1;
    var withinBrackets = function(idx) {
      if (noBrackets)
        return false;

      var previousOpening = selectors.lastIndexOf('(', idx);
      var previousClosing = selectors.lastIndexOf(')', idx);

      if (previousOpening == -1)
        return false;
      if (previousClosing > 0 && previousClosing < idx)
        return false;

      return true;
    };

    while (true) {
      var nextComma = selectors.indexOf(',', cursor + 1);
      var selector;

      if (nextComma === -1) {
        nextComma = selectors.length;
      } else if (withinBrackets(nextComma)) {
        cursor = nextComma + 1;
        continue;
      }
      selector = selectors.substring(lastComma, nextComma);
      lastComma = cursor = nextComma + 1;

      if (plain.indexOf(selector) == -1)
        plain.push(selector);

      if (nextComma === selectors.length)
        break;
    }

    return plain.sort().join(',');
  };

  var isSpecial = function(selector) {
    return specialSelectors[options.compatibility || '*'].test(selector);
  };

  var removeDuplicates = function(tokens) {
    var matched = {};
    var forRemoval = [];

    for (var i = 0, l = tokens.length; i < l; i++) {
      var token = tokens[i];
      if (typeof token == 'string' || token.block)
        continue;

      var id = token.body + '@' + token.selector;
      var alreadyMatched = matched[id];

      if (alreadyMatched) {
        forRemoval.push(alreadyMatched[0]);
        alreadyMatched.unshift(i);
      } else {
        matched[id] = [i];
      }
    }

    forRemoval = forRemoval.sort(function(a, b) {
      return a > b ? 1 : -1;
    });

    for (var j = 0, n = forRemoval.length; j < n; j++) {
      tokens.splice(forRemoval[j] - j, 1);
    }

    minificationsMade.unshift(forRemoval.length > 0);
  };

  var mergeAdjacent = function(tokens) {
    var forRemoval = [];
    var lastToken = { selector: null, body: null };

    for (var i = 0, l = tokens.length; i < l; i++) {
      var token = tokens[i];

      if (typeof token == 'string' || token.block)
        continue;

      if (token.selector == lastToken.selector) {
        var joinAt = [lastToken.body.split(';').length];
        lastToken.body = propertyOptimizer.process(lastToken.body + ';' + token.body, joinAt, false, token.selector);
        forRemoval.push(i);
      } else if (token.body == lastToken.body && !isSpecial(token.selector) && !isSpecial(lastToken.selector)) {
        lastToken.selector = cleanUpSelector(lastToken.selector + ',' + token.selector);
        forRemoval.push(i);
      } else {
        lastToken = token;
      }
    }

    for (var j = 0, m = forRemoval.length; j < m; j++) {
      tokens.splice(forRemoval[j] - j, 1);
    }

    minificationsMade.unshift(forRemoval.length > 0);
  };

  var reduceNonAdjacent = function(tokens) {
    var candidates = {};
    var moreThanOnce = [];

    for (var i = tokens.length - 1; i >= 0; i--) {
      var token = tokens[i];

      if (typeof token == 'string' || token.block)
        continue;

      var complexSelector = token.selector;
      var selectors = complexSelector.indexOf(',') > -1 && !isSpecial(complexSelector) ?
        complexSelector.split(',').concat(complexSelector) : // simplification, as :not() can have commas too
        [complexSelector];

      for (var j = 0, m = selectors.length; j < m; j++) {
        var selector = selectors[j];

        if (!candidates[selector])
          candidates[selector] = [];
        else
          moreThanOnce.push(selector);

        candidates[selector].push({
          where: i,
          partial: selector != complexSelector
        });
      }
    }

    var reducedInSimple = _reduceSimpleNonAdjacentCases(tokens, moreThanOnce, candidates);
    var reducedInComplex = _reduceComplexNonAdjacentCases(tokens, candidates);

    minificationsMade.unshift(reducedInSimple || reducedInComplex);
  };

  var _reduceSimpleNonAdjacentCases = function(tokens, matches, positions) {
    var reduced = false;

    for (var i = 0, l = matches.length; i < l; i++) {
      var selector = matches[i];
      var data = positions[selector];

      if (data.length < 2)
        continue;

      /* jshint loopfunc: true */
      _reduceSelector(tokens, selector, data, {
        filterOut: function(idx, bodies) {
          return data[idx].partial && bodies.length === 0;
        },
        callback: function(token, newBody, processedCount, tokenIdx) {
          if (!data[processedCount - tokenIdx - 1].partial) {
            token.body = newBody.join(';');
            reduced = true;
          }
        }
      });
    }

    return reduced;
  };

  var _reduceComplexNonAdjacentCases = function(tokens, positions) {
    var reduced = false;

    allSelectors:
    for (var complexSelector in positions) {
      if (complexSelector.indexOf(',') == -1) // simplification, as :not() can have commas too
        continue;

      var intoPosition = positions[complexSelector].pop().where;
      var intoToken = tokens[intoPosition];

      var selectors = isSpecial(complexSelector) ?
        [complexSelector] :
        complexSelector.split(',');
      var reducedBodies = [];

      for (var j = 0, m = selectors.length; j < m; j++) {
        var selector = selectors[j];
        var data = positions[selector];

        if (data.length < 2)
          continue allSelectors;

        /* jshint loopfunc: true */
        _reduceSelector(tokens, selector, data, {
          filterOut: function(idx) {
            return data[idx].where < intoPosition;
          },
          callback: function(token, newBody, processedCount, tokenIdx) {
            if (tokenIdx === 0)
              reducedBodies.push(newBody.join(';'));
          }
        });

        if (reducedBodies[reducedBodies.length - 1] != reducedBodies[0])
          continue allSelectors;
      }

      intoToken.body = reducedBodies[0];
      reduced = true;
    }

    return reduced;
  };

  var _reduceSelector = function(tokens, selector, data, options) {
    var bodies = [];
    var joinsAt = [];
    var splitBodies = [];
    var processedTokens = [];

    for (var j = data.length - 1, m = 0; j >= 0; j--) {
      if (options.filterOut(j, bodies))
        continue;

      var where = data[j].where;
      var token = tokens[where];
      var body = token.body;
      bodies.push(body);
      splitBodies.push(body.split(';'));
      processedTokens.push(where);
    }

    for (j = 0, m = bodies.length; j < m; j++) {
      if (bodies[j].length > 0)
        joinsAt.push((joinsAt[j - 1] || 0) + splitBodies[j].length);
    }

    var optimizedBody = propertyOptimizer.process(bodies.join(';'), joinsAt, true, selector);
    var optimizedProperties = optimizedBody.split(';');

    var processedCount = processedTokens.length;
    var propertyIdx = optimizedProperties.length - 1;
    var tokenIdx = processedCount - 1;

    while (tokenIdx >= 0) {
      if ((tokenIdx === 0 || splitBodies[tokenIdx].indexOf(optimizedProperties[propertyIdx]) > -1) && propertyIdx > -1) {
        propertyIdx--;
        continue;
      }

      var newBody = optimizedProperties.splice(propertyIdx + 1);
      options.callback(tokens[processedTokens[tokenIdx]], newBody, processedCount, tokenIdx);

      tokenIdx--;
    }
  };

  var optimize = function(tokens) {
    var noChanges = function() {
      return minificationsMade.length > 4 &&
        minificationsMade[0] === false &&
        minificationsMade[1] === false;
    };

    tokens = Array.isArray(tokens) ? tokens : [tokens];
    for (var i = 0, l = tokens.length; i < l; i++) {
      var token = tokens[i];

      if (token.selector) {
        token.selector = cleanUpSelector(token.selector);
        token.body = propertyOptimizer.process(token.body, false, false, token.selector);
      } else if (token.block) {
        optimize(token.body);
      }
    }

    // Run until 2 last operations do not yield any changes
    minificationsMade = [];
    while (true) {
      if (noChanges())
        break;
      removeDuplicates(tokens);

      if (noChanges())
        break;
      mergeAdjacent(tokens);

      if (noChanges())
        break;
      reduceNonAdjacent(tokens);
    }
  };

  var rebuild = function(tokens) {
    var rebuilt = [];

    tokens = Array.isArray(tokens) ? tokens : [tokens];
    for (var i = 0, l = tokens.length; i < l; i++) {
      var token = tokens[i];

      if (typeof token == 'string') {
        rebuilt.push(token);
        continue;
      }

      var name = token.block || token.selector;
      var body = token.block ? rebuild(token.body) : token.body;

      if (body.length > 0)
        rebuilt.push(name + '{' + body + '}');
    }

    return rebuilt.join(options.keepBreaks ? options.lineBreak : '');
  };

  return {
    process: function() {
      var tokenized = new Tokenizer(data, context).process();
      optimize(tokenized);
      return rebuild(tokenized);
    }
  };
};
