var CleanUp = require('./clean-up');
var Splitter = require('../../utils/splitter');

var RGB = require('../../colors/rgb');
var HSL = require('../../colors/hsl');
var HexNameShortener = require('../../colors/hex-name-shortener');

var DEFAULT_ROUNDING_PRECISION = 2;
var CHARSET_TOKEN = '@charset';
var CHARSET_REGEXP = new RegExp('^' + CHARSET_TOKEN, 'i');

function SimpleOptimizer(options) {
  this.options = options;

  var units = ['px', 'em', 'ex', 'cm', 'mm', 'in', 'pt', 'pc', '%'];
  if (options.compatibility.units.rem)
    units.push('rem');
  options.unitsRegexp = new RegExp('(^|\\s|\\(|,)0(?:' + units.join('|') + ')', 'g');

  options.precision = {};
  options.precision.value = options.roundingPrecision === undefined ?
    DEFAULT_ROUNDING_PRECISION :
    options.roundingPrecision;
  options.precision.multiplier = Math.pow(10, options.precision.value);
  options.precision.regexp = new RegExp('(\\d*\\.\\d{' + (options.precision.value + 1) + ',})px', 'g');

  options.updateMetadata = this.options.advanced;
}

var valueMinifiers = {
  'background': function (value) {
    return value == 'none' || value == 'transparent' ? '0 0' : value;
  },
  'border-*-radius': function (value) {
    if (value.indexOf('/') == -1)
      return value;

    var parts = value.split(/\s*\/\s*/);
    if (parts[0] == parts[1])
      return parts[0];
    else
      return parts[0] + '/' + parts[1];
  },
  'filter': function (value) {
    if (value.indexOf('DXImageTransform') === value.lastIndexOf('DXImageTransform')) {
      value = value.replace(/progid:DXImageTransform\.Microsoft\.(Alpha|Chroma)/, function (match, filter) {
        return filter.toLowerCase();
      });
    }

    return value
      .replace(/,(\S)/g, ', $1')
      .replace(/ ?= ?/g, '=');
  },
  'font': function (value) {
    var parts = value.split(' ');

    if (parts[1] != 'normal' && parts[1] != 'bold' && !/^[1-9]00/.test(parts[1]))
      parts[0] = this['font-weight'](parts[0]);

    return parts.join(' ');
  },
  'font-weight': function (value) {
    if (value == 'normal')
      return '400';
    else if (value == 'bold')
      return '700';
    else
      return value;
  },
  'outline': function (value) {
    return value == 'none' ? '0' : value;
  }
};

function zeroMinifier(_, value) {
  if (value.indexOf('0') == -1)
    return value;

  return value
    .replace(/\-0$/g, '0')
    .replace(/\-0([^\.])/g, '0$1')
    .replace(/(^|\s)0+([1-9])/g, '$1$2')
    .replace(/(^|\D)\.0+(\D|$)/g, '$10$2')
    .replace(/(^|\D)\.0+(\D|$)/g, '$10$2')
    .replace(/\.([1-9]*)0+(\D|$)/g, function(match, nonZeroPart, suffix) {
      return (nonZeroPart.length > 0 ? '.' : '') + nonZeroPart + suffix;
    })
    .replace(/(^|\D)0\.(\d)/g, '$1.$2');
}

function precisionMinifier(_, value, precisionOptions) {
  if (precisionOptions.value === -1 || value.indexOf('.') === -1)
    return value;

  return value
    .replace(precisionOptions.regexp, function(match, number) {
      return Math.round(parseFloat(number) * precisionOptions.multiplier) / precisionOptions.multiplier + 'px';
    })
    .replace(/(\d)\.($|\D)/g, '$1$2');
}

function unitMinifier(_, value, unitsRegexp) {
  return value.replace(unitsRegexp, '$1' + '0');
}

function multipleZerosMinifier(property, value) {
  if (value.indexOf('0 0 0 0') == -1)
    return value;

  if (property.indexOf('box-shadow') > -1)
    return value == '0 0 0 0' ? '0 0' : value;

  return value.replace(/^0 0 0 0$/, '0');
}

function colorMininifier(property, value, compatibility) {
  if (value.indexOf('#') === -1 && value.indexOf('rgb') == -1 && value.indexOf('hsl') == -1)
    return HexNameShortener.shorten(value);

  value = value
    .replace(/rgb\((\-?\d+),(\-?\d+),(\-?\d+)\)/g, function (match, red, green, blue) {
      return new RGB(red, green, blue).toHex();
    })
    .replace(/hsl\((-?\d+),(-?\d+)%?,(-?\d+)%?\)/g, function (match, hue, saturation, lightness) {
      return new HSL(hue, saturation, lightness).toHex();
    })
    .replace(/(^|[^='"])#([0-9a-f]{6})/gi, function (match, prefix, color) {
      if (color[0] == color[1] && color[2] == color[3] && color[4] == color[5])
        return prefix + '#' + color[0] + color[2] + color[4];
      else
        return prefix + '#' + color;
    })
    .replace(/(rgb|rgba|hsl|hsla)\(([^\)]+)\)/g, function(match, colorFunction, colorDef) {
      var tokens = colorDef.split(',');
      var applies = colorFunction == 'hsl' || colorFunction == 'hsla' || tokens[0].indexOf('%') > -1;
      if (!applies)
        return match;

      if (tokens[1].indexOf('%') == -1)
        tokens[1] += '%';
      if (tokens[2].indexOf('%') == -1)
        tokens[2] += '%';
      return colorFunction + '(' + tokens.join(',') + ')';
    });

  if (compatibility.colors.opacity) {
    value = value.replace(/(?:rgba|hsla)\(0,0%?,0%?,0\)/g, function (match) {
      if (new Splitter(',').split(value).pop().indexOf('gradient(') > -1)
        return match;

      return 'transparent';
    });
  }

  return HexNameShortener.shorten(value);
}

function reduce(body, options) {
  var reduced = [];
  var properties = [];
  var newProperty;

  for (var i = 0, l = body.length; i < l; i++) {
    var token = body[i];

    // FIXME: the check should be gone with #396
    if (token.value.indexOf('__ESCAPED_') === 0) {
      reduced.push(token);
      properties.push(token.value);
      continue;
    }

    var firstColon = token.value.indexOf(':');
    var property = token.value.substring(0, firstColon);
    var value = token.value.substring(firstColon + 1);
    var important = false;

    if (!options.compatibility.properties.iePrefixHack && (property[0] == '_' || property[0] == '*'))
      continue;

    if (value.indexOf('!important') > 0 || value.indexOf('! important') > 0) {
      value = value.substring(0, value.indexOf('!')).trim();
      important = true;
    }

    if (property.indexOf('border') === 0 && property.indexOf('radius') > 0)
      value = valueMinifiers['border-*-radius'](value);

    if (valueMinifiers[property])
      value = valueMinifiers[property](value);

    value = precisionMinifier(property, value, options.precision);
    value = zeroMinifier(property, value);
    value = unitMinifier(property, value, options.unitsRegexp);
    value = multipleZerosMinifier(property, value);
    value = colorMininifier(property, value, options.compatibility);

    newProperty = property + ':' + value + (important ? '!important' : '');
    reduced.push({ value: newProperty, metadata: token.metadata });
    properties.push(newProperty);
  }

  return {
    tokenized: reduced,
    list: properties
  };
}

SimpleOptimizer.prototype.optimize = function(tokens) {
  var self = this;
  var hasCharset = false;
  var options = this.options;

  function _optimize(tokens) {
    for (var i = 0, l = tokens.length; i < l; i++) {
      var token = tokens[i];
      // FIXME: why it's so?
      if (!token)
        break;

      if (token.kind == 'selector') {
        var newSelectors = CleanUp.selectors(token.value, !options.compatibility.selectors.ie7Hack);
        token.value = newSelectors.tokenized;

        if (token.value.length === 0) {
          tokens.splice(i, 1);
          i--;
          continue;
        }
        var newBody = reduce(token.body, self.options);
        token.body = newBody.tokenized;

        if (options.updateMetadata) {
          token.metadata.body = newBody.list.join(';');
          token.metadata.bodiesList = newBody.list;
          token.metadata.selector = newSelectors.list.join(',');
          token.metadata.selectorsList = newSelectors.list;
        }
      } else if (token.kind == 'block') {
        token.value = CleanUp.block(token.value);
        if (token.isFlatBlock)
          token.body = reduce(token.body, self.options).tokenized;
        else
          _optimize(token.body);
      } else if (token.kind == 'at-rule') {
        token.value = CleanUp.atRule(token.value);

        if (CHARSET_REGEXP.test(token.value)) {
          if (hasCharset || token.value.indexOf(CHARSET_TOKEN) == -1) {
            tokens.splice(i, 1);
            i--;
          } else {
            hasCharset = true;
            tokens.splice(i, 1);
            tokens.unshift({ kind: 'at-rule', value: token.value.replace(CHARSET_REGEXP, CHARSET_TOKEN) });
          }
        }
      }
    }
  }

  _optimize(tokens);
};

module.exports = SimpleOptimizer;
