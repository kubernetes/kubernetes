
// Validates various CSS property values

var Splitter = require('../text/splitter');

module.exports = (function () {
  // Regexes used for stuff
  var widthKeywords = ['thin', 'thick', 'medium', 'inherit', 'initial'];
  var cssUnitRegexStr = '(\\-?\\.?\\d+\\.?\\d*(px|%|em|rem|in|cm|mm|ex|pt|pc|vw|vh|vmin|vmax|)|auto|inherit)';
  var cssCalcRegexStr = '(\\-moz\\-|\\-webkit\\-)?calc\\([^\\)]+\\)';
  var cssFunctionNoVendorRegexStr = '[A-Z]+(\\-|[A-Z]|[0-9])+\\(([A-Z]|[0-9]|\\ |\\,|\\#|\\+|\\-|\\%|\\.|\\(|\\))*\\)';
  var cssFunctionVendorRegexStr = '\\-(\\-|[A-Z]|[0-9])+\\(([A-Z]|[0-9]|\\ |\\,|\\#|\\+|\\-|\\%|\\.|\\(|\\))*\\)';
  var cssVariableRegexStr = 'var\\(\\-\\-[^\\)]+\\)';
  var cssFunctionAnyRegexStr = '(' + cssVariableRegexStr + '|' + cssFunctionNoVendorRegexStr + '|' + cssFunctionVendorRegexStr + ')';
  var cssUnitOrCalcRegexStr = '(' + cssUnitRegexStr + '|' + cssCalcRegexStr + ')';
  var cssUnitAnyRegexStr = '(none|' + widthKeywords.join('|') + '|' + cssUnitRegexStr + '|' + cssVariableRegexStr + '|' + cssFunctionNoVendorRegexStr + '|' + cssFunctionVendorRegexStr + ')';

  var cssFunctionNoVendorRegex = new RegExp('^' + cssFunctionNoVendorRegexStr + '$', 'i');
  var cssFunctionVendorRegex = new RegExp('^' + cssFunctionVendorRegexStr + '$', 'i');
  var cssVariableRegex = new RegExp('^' + cssVariableRegexStr + '$', 'i');
  var cssFunctionAnyRegex = new RegExp('^' + cssFunctionAnyRegexStr + '$', 'i');
  var cssUnitRegex = new RegExp('^' + cssUnitRegexStr + '$', 'i');
  var cssUnitOrCalcRegex = new RegExp('^' + cssUnitOrCalcRegexStr + '$', 'i');
  var cssUnitAnyRegex = new RegExp('^' + cssUnitAnyRegexStr + '$', 'i');

  var backgroundRepeatKeywords = ['repeat', 'no-repeat', 'repeat-x', 'repeat-y', 'inherit'];
  var backgroundAttachmentKeywords = ['inherit', 'scroll', 'fixed', 'local'];
  var backgroundPositionKeywords = ['center', 'top', 'bottom', 'left', 'right'];
  var backgroundSizeKeywords = ['contain', 'cover'];
  var listStyleTypeKeywords = ['armenian', 'circle', 'cjk-ideographic', 'decimal', 'decimal-leading-zero', 'disc', 'georgian', 'hebrew', 'hiragana', 'hiragana-iroha', 'inherit', 'katakana', 'katakana-iroha', 'lower-alpha', 'lower-greek', 'lower-latin', 'lower-roman', 'none', 'square', 'upper-alpha', 'upper-latin', 'upper-roman'];
  var listStylePositionKeywords = ['inside', 'outside', 'inherit'];
  var outlineStyleKeywords = ['auto', 'inherit', 'hidden', 'none', 'dotted', 'dashed', 'solid', 'double', 'groove', 'ridge', 'inset', 'outset'];

  var validator = {
    isValidHexColor: function (s) {
      return (s.length === 4 || s.length === 7) && s[0] === '#';
    },
    isValidRgbaColor: function (s) {
      s = s.split(' ').join('');
      return s.length > 0 && s.indexOf('rgba(') === 0 && s.indexOf(')') === s.length - 1;
    },
    isValidHslaColor: function (s) {
      s = s.split(' ').join('');
      return s.length > 0 && s.indexOf('hsla(') === 0 && s.indexOf(')') === s.length - 1;
    },
    isValidNamedColor: function (s) {
      // We don't really check if it's a valid color value, but allow any letters in it
      return s !== 'auto' && (s === 'transparent' || s === 'inherit' || /^[a-zA-Z]+$/.test(s));
    },
    isValidVariable: function(s) {
      return cssVariableRegex.test(s);
    },
    isValidColor: function (s) {
      return validator.isValidNamedColor(s) || validator.isValidHexColor(s) || validator.isValidRgbaColor(s) || validator.isValidHslaColor(s) || validator.isValidVariable(s);
    },
    isValidUrl: function (s) {
      // NOTE: at this point all URLs are replaced with placeholders by clean-css, so we check for those placeholders
      return s.indexOf('__ESCAPED_URL_CLEAN_CSS') === 0;
    },
    isValidUnit: function (s) {
      return cssUnitAnyRegex.test(s);
    },
    isValidUnitWithoutFunction: function (s) {
      return cssUnitRegex.test(s);
    },
    isValidFunctionWithoutVendorPrefix: function (s) {
      return cssFunctionNoVendorRegex.test(s);
    },
    isValidFunctionWithVendorPrefix: function (s) {
      return cssFunctionVendorRegex.test(s);
    },
    isValidFunction: function (s) {
      return cssFunctionAnyRegex.test(s);
    },
    isValidBackgroundRepeat: function (s) {
      return backgroundRepeatKeywords.indexOf(s) >= 0 || validator.isValidVariable(s);
    },
    isValidBackgroundAttachment: function (s) {
      return backgroundAttachmentKeywords.indexOf(s) >= 0 || validator.isValidVariable(s);
    },
    isValidBackgroundPositionPart: function (s) {
      return backgroundPositionKeywords.indexOf(s) >= 0 || cssUnitOrCalcRegex.test(s) || validator.isValidVariable(s);
    },
    isValidBackgroundPosition: function (s) {
      if (s === 'inherit')
        return true;

      var parts = s.split(' ');
      for (var i = 0, l = parts.length; i < l; i++) {
        if (parts[i] === '')
          continue;
        if (validator.isValidBackgroundPositionPart(parts[i]) || validator.isValidVariable(parts[i]))
          continue;

        return false;
      }

      return true;
    },
    isValidBackgroundSizePart: function(s) {
      return backgroundSizeKeywords.indexOf(s) >= 0 || cssUnitRegex.test(s) || validator.isValidVariable(s);
    },
    isValidBackgroundPositionAndSize: function(s) {
      if (s.indexOf('/') < 0)
        return false;

      var twoParts = new Splitter('/').split(s);
      return validator.isValidBackgroundSizePart(twoParts.pop()) && validator.isValidBackgroundPositionPart(twoParts.pop());
    },
    isValidListStyleType: function (s) {
      return listStyleTypeKeywords.indexOf(s) >= 0 || validator.isValidVariable(s);
    },
    isValidListStylePosition: function (s) {
      return listStylePositionKeywords.indexOf(s) >= 0 || validator.isValidVariable(s);
    },
    isValidOutlineColor: function (s) {
      return s === 'invert' || validator.isValidColor(s) || validator.isValidVendorPrefixedValue(s);
    },
    isValidOutlineStyle: function (s) {
      return outlineStyleKeywords.indexOf(s) >= 0 || validator.isValidVariable(s);
    },
    isValidOutlineWidth: function (s) {
      return validator.isValidUnit(s) || widthKeywords.indexOf(s) >= 0 || validator.isValidVariable(s);
    },
    isValidVendorPrefixedValue: function (s) {
      return /^-([A-Za-z0-9]|-)*$/gi.test(s);
    },
    areSameFunction: function (a, b) {
      if (!validator.isValidFunction(a) || !validator.isValidFunction(b))
        return false;

      var f1name = a.substring(0, a.indexOf('('));
      var f2name = b.substring(0, b.indexOf('('));

      return f1name === f2name;
    }
  };

  return validator;
})();
