
// Helper for tokenizing the contents of a CSS selector block

module.exports = (function() {
  var createTokenPrototype = function (processable) {
    var important = '!important';

    // Constructor for tokens
    function Token (prop, p2, p3) {
      this.prop = prop;
      if (typeof(p2) === 'string') {
        this.value = p2;
        this.isImportant = p3;
      }
      else {
        this.value = processable[prop].defaultValue;
        this.isImportant = p2;
      }
    }

    Token.prototype.prop = null;
    Token.prototype.value = null;
    Token.prototype.granularValues = null;
    Token.prototype.components = null;
    Token.prototype.position = null;
    Token.prototype.isImportant = false;
    Token.prototype.isDirty = false;
    Token.prototype.isShorthand = false;
    Token.prototype.isIrrelevant = false;
    Token.prototype.isReal = true;
    Token.prototype.isMarkedForDeletion = false;
    Token.prototype.metadata = null;

    // Tells if this token is a component of the other one
    Token.prototype.isComponentOf = function (other) {
      if (!processable[this.prop] || !processable[other.prop])
        return false;
      if (!(processable[other.prop].components instanceof Array) || !processable[other.prop].components.length)
        return false;

      return processable[other.prop].components.indexOf(this.prop) >= 0;
    };

    // Clones a token
    Token.prototype.clone = function (isImportant) {
      var token = new Token(this.prop, this.value, (typeof(isImportant) !== 'undefined' ? isImportant : this.isImportant));
      return token;
    };

    // Creates an irrelevant token with the same prop
    Token.prototype.cloneIrrelevant = function (isImportant) {
      var token = Token.makeDefault(this.prop, (typeof(isImportant) !== 'undefined' ? isImportant : this.isImportant));
      token.isIrrelevant = true;
      return token;
    };

    // Creates an array of property tokens with their default values
    Token.makeDefaults = function (props, important) {
      return props.map(function(prop) {
        return new Token(prop, important);
      });
    };

    // Parses one CSS property declaration into a token
    Token.tokenizeOne = function (fullProp) {
      // Find first colon
      var colonPos = fullProp.value.indexOf(':');

      if (colonPos < 0) {
        // This property doesn't have a colon, it's invalid. Let's keep it intact anyway.
        return new Token('', fullProp.value);
      }

      // Parse parts of the property
      var prop = fullProp.value.substr(0, colonPos).trim();
      var value = fullProp.value.substr(colonPos + 1).trim();
      var isImportant = false;
      var importantPos = value.indexOf(important);

      // Check if the property is important
      if (importantPos >= 1 && importantPos === value.length - important.length) {
        value = value.substr(0, importantPos).trim();
        isImportant = true;
      }

      // Return result
      var result = new Token(prop, value, isImportant);

      // If this is a shorthand, break up its values
      // NOTE: we need to do this for all shorthands because otherwise we couldn't remove default values from them
      if (processable[prop] && processable[prop].isShorthand) {
        result.isShorthand = true;
        result.components = processable[prop].breakUp(result);
        result.isDirty = true;
      }

      result.metadata = fullProp.metadata;

      return result;
    };

    // Breaks up a string of CSS property declarations into tokens so that they can be handled more easily
    Token.tokenize = function (input) {
      // Split the input by semicolons and parse the parts
      var tokens = input.map(Token.tokenizeOne);
      return tokens;
    };

    // Transforms tokens back into CSS properties
    Token.detokenize = function (tokens) {
      // If by mistake the input is not an array, make it an array
      if (!(tokens instanceof Array)) {
        tokens = [tokens];
      }

      var tokenized = [];
      var list = [];

      // This step takes care of putting together the components of shorthands
      // NOTE: this is necessary to do for every shorthand, otherwise we couldn't remove their default values
      for (var i = 0; i < tokens.length; i++) {
        var t = tokens[i];
        if (t.isShorthand && t.isDirty) {
          var news = processable[t.prop].putTogether(t.prop, t.components, t.isImportant);
          Array.prototype.splice.apply(tokens, [i, 1].concat(news));
          t.isDirty = false;
          i--;
          continue;
        }

        // FIXME: the check should be gone with #396
        var property = t.prop === '' && t.value.indexOf('__ESCAPED_') === 0 ?
          t.value :
          t.prop + ':' + t.value + (t.isImportant ? important : '');
        tokenized.push({ value: property, metadata: t.metadata || {} });
        list.push(property);
      }

      return {
        list: list,
        tokenized: tokenized
      };
    };

    // Gets the final (detokenized) length of the given tokens
    Token.getDetokenizedLength = function (tokens) {
      // If by mistake the input is not an array, make it an array
      if (!(tokens instanceof Array)) {
        tokens = [tokens];
      }

      var result = 0;

      // This step takes care of putting together the components of shorthands
      // NOTE: this is necessary to do for every shorthand, otherwise we couldn't remove their default values
      for (var i = 0; i < tokens.length; i++) {
        var t = tokens[i];
        if (t.isShorthand && t.isDirty) {
          var news = processable[t.prop].putTogether(t.prop, t.components, t.isImportant);
          Array.prototype.splice.apply(tokens, [i, 1].concat(news));
          t.isDirty = false;
          i--;
          continue;
        }

        if (t.prop) {
          result += t.prop.length + 1;
        }
        if (t.value) {
          result += t.value.length;
        }
        if (t.isImportant) {
          result += important.length;
        }
      }

      return result;
    };

    return Token;
  };

  return {
    createTokenPrototype: createTokenPrototype
  };

})();
