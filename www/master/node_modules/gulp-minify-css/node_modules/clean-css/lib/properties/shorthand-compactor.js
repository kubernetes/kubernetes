
// Compacts the tokens by transforming properties into their shorthand notations when possible

module.exports = (function () {
  var isHackValue = function (t) { return t.value === '__hack'; };

  var compactShorthands = function(tokens, isImportant, processable, Token) {
    // Contains the components found so far, grouped by shorthand name
    var componentsSoFar = { };

    // Initializes a prop in componentsSoFar
    var initSoFar = function (shprop, last, clearAll) {
      var found = {};
      var shorthandPosition;

      if (!clearAll && componentsSoFar[shprop]) {
        for (var i = 0; i < processable[shprop].components.length; i++) {
          var prop = processable[shprop].components[i];
          found[prop] = [];

          if (!(componentsSoFar[shprop].found[prop]))
            continue;

          for (var ii = 0; ii < componentsSoFar[shprop].found[prop].length; ii++) {
            var comp = componentsSoFar[shprop].found[prop][ii];

            if (comp.isMarkedForDeletion)
              continue;

            found[prop].push(comp);

            if (comp.position && (!shorthandPosition || comp.position < shorthandPosition))
              shorthandPosition = comp.position;
          }
        }
      }
      componentsSoFar[shprop] = {
        lastShorthand: last,
        found: found,
        shorthandPosition: shorthandPosition
      };
    };

    // Adds a component to componentsSoFar
    var addComponentSoFar = function (token, index) {
      var shprop = processable[token.prop].componentOf;
      if (!componentsSoFar[shprop])
        initSoFar(shprop);
      if (!componentsSoFar[shprop].found[token.prop])
        componentsSoFar[shprop].found[token.prop] = [];

      // Add the newfound component to componentsSoFar
      componentsSoFar[shprop].found[token.prop].push(token);

      if (!componentsSoFar[shprop].shorthandPosition && index) {
        // If the haven't decided on where the shorthand should go, put it in the place of this component
        componentsSoFar[shprop].shorthandPosition = index;
      }
    };

    // Tries to compact a prop in componentsSoFar
    var compactSoFar = function (prop) {
      var i;
      var componentsCount = processable[prop].components.length;

      // Check basics
      if (!componentsSoFar[prop] || !componentsSoFar[prop].found)
        return false;

      // Find components for the shorthand
      var components = [];
      var realComponents = [];
      for (i = 0 ; i < componentsCount; i++) {
        // Get property name
        var pp = processable[prop].components[i];

        if (componentsSoFar[prop].found[pp] && componentsSoFar[prop].found[pp].length) {
          // We really found it
          var foundRealComp = componentsSoFar[prop].found[pp][0];
          components.push(foundRealComp);
          if (foundRealComp.isReal !== false) {
            realComponents.push(foundRealComp);
          }
        } else if (componentsSoFar[prop].lastShorthand) {
          // It's defined in the previous shorthand
          var c = componentsSoFar[prop].lastShorthand.components[i].clone(isImportant);
          components.push(c);
        } else {
          // Couldn't find this component at all
          return false;
        }
      }

      if (realComponents.length === 0) {
        // Couldn't find enough components, sorry
        return false;
      }

      if (realComponents.length === componentsCount) {
        // When all the components are from real values, only allow shorthanding if their understandability allows it
        // This is the case when every component can override their default values, or when all of them use the same function

        var canOverrideDefault = true;
        var functionNameMatches = true;
        var functionName;

        for (var ci = 0; ci < realComponents.length; ci++) {
          var rc = realComponents[ci];

          if (!processable[rc.prop].canOverride(processable[rc.prop].defaultValue, rc.value)) {
            canOverrideDefault = false;
          }
          var iop = rc.value.indexOf('(');
          if (iop >= 0) {
            var otherFunctionName = rc.value.substring(0, iop);
            if (functionName)
              functionNameMatches = functionNameMatches && otherFunctionName === functionName;
            else
              functionName = otherFunctionName;
          }
        }

        if (!canOverrideDefault || !functionNameMatches)
          return false;
      }

      // Compact the components into a shorthand
      var compacted = processable[prop].putTogether(prop, components, isImportant);
      if (!(compacted instanceof Array)) {
        compacted = [compacted];
      }

      var compactedLength = Token.getDetokenizedLength(compacted);
      var authenticLength = Token.getDetokenizedLength(realComponents);

      if (realComponents.length === componentsCount || compactedLength < authenticLength || components.some(isHackValue)) {
        compacted[0].isShorthand = true;
        compacted[0].components = processable[prop].breakUp(compacted[0]);

        // Mark the granular components for deletion
        for (i = 0; i < realComponents.length; i++) {
          realComponents[i].isMarkedForDeletion = true;
        }

        // Mark the position of the new shorthand
        tokens[componentsSoFar[prop].shorthandPosition].replaceWith = compacted;

        // Reinitialize the thing for further compacting
        initSoFar(prop, compacted[0]);
        for (i = 1; i < compacted.length; i++) {
          addComponentSoFar(compacted[i]);
        }

        // Yes, we can keep the new shorthand!
        return true;
      }

      return false;
    };

    // Tries to compact all properties currently in componentsSoFar
    var compactAllSoFar = function () {
      for (var i in componentsSoFar) {
        if (componentsSoFar.hasOwnProperty(i)) {
          while (compactSoFar(i)) { }
        }
      }
    };

    var i, token;

    // Go through each token and collect components for each shorthand as we go on
    for (i = 0; i < tokens.length; i++) {
      token = tokens[i];
      if (token.isMarkedForDeletion) {
        continue;
      }
      if (!processable[token.prop]) {
        // We don't know what it is, move on
        continue;
      }
      if (processable[token.prop].isShorthand) {
        // Found an instance of a full shorthand
        // NOTE: we should NOT mix together tokens that come before and after the shorthands

        if (token.isImportant === isImportant || (token.isImportant && !isImportant)) {
          // Try to compact what we've found so far
          while (compactSoFar(token.prop)) { }
          // Reset
          initSoFar(token.prop, token, true);
        }

        // TODO: when the old optimizer is removed, take care of this corner case:
        //   div{background-color:#111;background-image:url(aaa);background:linear-gradient(aaa);background-repeat:no-repeat;background-position:1px 2px;background-attachment:scroll}
        //   -> should not be shorthanded / minified at all because the result wouldn't be equivalent to the original in any browser
      } else if (processable[token.prop].componentOf) {
        // Found a component of a shorthand
        if (token.isImportant === isImportant) {
          // Same importantness
          token.position = i;
          addComponentSoFar(token, i);
        } else if (!isImportant && token.isImportant) {
          // Use importants for optimalization opportunities
          // https://github.com/jakubpawlowicz/clean-css/issues/184
          var importantTrickComp = new Token(token.prop, token.value, isImportant);
          importantTrickComp.isIrrelevant = true;
          importantTrickComp.isReal = false;
          addComponentSoFar(importantTrickComp);
        }
      } else {
        // This is not a shorthand and not a component, don't care about it
        continue;
      }
    }

    // Perform all possible compactions
    compactAllSoFar();

    // Process the results - throw away stuff marked for deletion, insert compacted things, etc.
    var result = [];
    for (i = 0; i < tokens.length; i++) {
      token = tokens[i];

      if (token.replaceWith) {
        for (var ii = 0; ii < token.replaceWith.length; ii++) {
          result.push(token.replaceWith[ii]);
        }
      }
      if (!token.isMarkedForDeletion) {
        result.push(token);
      }

      token.isMarkedForDeletion = false;
      token.replaceWith = null;
    }

    return result;
  };

  return {
    compactShorthands: compactShorthands
  };

})();
