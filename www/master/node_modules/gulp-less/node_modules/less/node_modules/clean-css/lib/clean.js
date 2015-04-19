/**
 * Clean-css - https://github.com/GoalSmashers/clean-css
 * Released under the terms of MIT license
 *
 * Copyright (C) 2011-2014 GoalSmashers.com
 */

var ColorShortener = require('./colors/shortener');
var ColorHSLToHex = require('./colors/hsl-to-hex');
var ColorRGBToHex = require('./colors/rgb-to-hex');
var ColorLongToShortHex = require('./colors/long-to-short-hex');

var ImportInliner = require('./imports/inliner');
var UrlRebase = require('./images/url-rebase');
var EmptyRemoval = require('./selectors/empty-removal');

var CommentsProcessor = require('./text/comments');
var ExpressionsProcessor = require('./text/expressions');
var FreeTextProcessor = require('./text/free');
var UrlsProcessor = require('./text/urls');
var NameQuotesProcessor = require('./text/name-quotes');
var Splitter = require('./text/splitter');

var SelectorsOptimizer = require('./selectors/optimizer');

var CleanCSS = module.exports = function CleanCSS(options) {
  options = options || {};

  // back compat
  if (!(this instanceof CleanCSS))
    return new CleanCSS(options);

  options.keepBreaks = options.keepBreaks || false;

  //active by default
  if (undefined === options.processImport)
    options.processImport = true;

  this.options = options;
  this.stats = {};
  this.context = {
    errors: [],
    warnings: [],
    debug: options.debug
  };
  this.errors = this.context.errors;
  this.warnings = this.context.warnings;
  this.lineBreak = process.platform == 'win32' ? '\r\n' : '\n';
};

CleanCSS.prototype.minify = function(data, callback) {
  var options = this.options;

  if (Buffer.isBuffer(data))
    data = data.toString();

  if (options.processImport || data.indexOf('@shallow') > 0) {
    // inline all imports
    var self = this;
    var runner = callback ?
      process.nextTick :
      function(callback) { return callback(); };

    return runner(function() {
      return new ImportInliner(self.context, options.inliner).process(data, {
        localOnly: !callback,
        root: options.root || process.cwd(),
        relativeTo: options.relativeTo,
        whenDone: function(data) {
          return minify.call(self, data, callback);
        }
      });
    });
  } else {
    return minify.call(this, data, callback);
  }
};

var minify = function(data, callback) {
  var startedAt;
  var stats = this.stats;
  var options = this.options;
  var context = this.context;
  var lineBreak = this.lineBreak;

  var commentsProcessor = new CommentsProcessor(
    context,
    'keepSpecialComments' in options ? options.keepSpecialComments : '*',
    options.keepBreaks,
    lineBreak
  );
  var expressionsProcessor = new ExpressionsProcessor();
  var freeTextProcessor = new FreeTextProcessor();
  var urlsProcessor = new UrlsProcessor(context);
  var nameQuotesProcessor = new NameQuotesProcessor();

  if (options.debug) {
    this.startedAt = process.hrtime();
    this.stats.originalSize = data.length;
  }

  var replace = function() {
    if (typeof arguments[0] == 'function')
      arguments[0]();
    else
      data = data.replace.apply(data, arguments);
  };

  // replace function
  if (options.benchmark) {
    var originalReplace = replace;
    replace = function(pattern, replacement) {
      var name = typeof pattern == 'function' ?
        /function (\w+)\(/.exec(pattern.toString())[1] :
        pattern;

      var start = process.hrtime();
      originalReplace(pattern, replacement);

      var itTook = process.hrtime(start);
      console.log('%d ms: ' + name, 1000 * itTook[0] + itTook[1] / 1000000);
    };
  }

  if (options.debug) {
    startedAt = process.hrtime();
    stats.originalSize = data.length;
  }

  replace(function escapeComments() {
    data = commentsProcessor.escape(data);
  });

  // replace all escaped line breaks
  replace(/\\(\r\n|\n)/gm, '');

  // strip parentheses in urls if possible (no spaces inside)
  replace(/url\((['"])([^\)]+)['"]\)/g, function(match, quote, url) {
    var unsafeDataURI = url.indexOf('data:') === 0 && url.match(/data:\w+\/[^;]+;base64,/) === null;
    if (url.match(/[ \t]/g) !== null || unsafeDataURI)
      return 'url(' + quote + url + quote + ')';
    else
      return 'url(' + url + ')';
  });

  // strip parentheses in animation & font names
  replace(function removeQuotes() {
    data = nameQuotesProcessor.process(data);
  });

  // strip parentheses in @keyframes
  replace(/@(\-moz\-|\-o\-|\-webkit\-)?keyframes ([^{]+)/g, function(match, prefix, name) {
    prefix = prefix || '';
    return '@' + prefix + 'keyframes ' + (name.indexOf(' ') > -1 ? name : name.replace(/['"]/g, ''));
  });

  // IE shorter filters, but only if single (IE 7 issue)
  replace(/progid:DXImageTransform\.Microsoft\.(Alpha|Chroma)(\([^\)]+\))([;}'"])/g, function(match, filter, args, suffix) {
    return filter.toLowerCase() + args + suffix;
  });

  replace(function escapeExpressions() {
    data = expressionsProcessor.escape(data);
  });

  // strip parentheses in attribute values
  replace(/\[([^\]]+)\]/g, function(match, content) {
    var eqIndex = content.indexOf('=');
    var singleQuoteIndex = content.indexOf('\'');
    var doubleQuoteIndex = content.indexOf('"');
    if (eqIndex < 0 && singleQuoteIndex < 0 && doubleQuoteIndex < 0)
      return match;
    if (singleQuoteIndex === 0 || doubleQuoteIndex === 0)
      return match;

    var key = content.substring(0, eqIndex);
    var value = content.substring(eqIndex + 1, content.length);

    if (/^['"](?:[a-zA-Z][a-zA-Z\d\-_]+)['"]$/.test(value))
      return '[' + key + '=' + value.substring(1, value.length - 1) + ']';
    else
      return match;
  });

  replace(function escapeFreeText() {
    data = freeTextProcessor.escape(data);
  });

  replace(function escapeUrls() {
    data = urlsProcessor.escape(data);
  });

  // remove invalid special declarations
  replace(/@charset [^;]+;/ig, function (match) {
    return match.indexOf('@charset') > -1 ? match : '';
  });

  // whitespace inside attribute selectors brackets
  replace(/\[([^\]]+)\]/g, function(match) {
    return match.replace(/\s/g, '');
  });

  // line breaks
  replace(/[\r]?\n/g, ' ');

  // multiple whitespace
  replace(/[\t ]+/g, ' ');

  // multiple semicolons (with optional whitespace)
  replace(/;[ ]?;+/g, ';');

  // multiple line breaks to one
  replace(/ (?:\r\n|\n)/g, lineBreak);
  replace(/(?:\r\n|\n)+/g, lineBreak);

  // remove spaces around selectors
  replace(/ ([+~>]) /g, '$1');

  // remove extra spaces inside content
  replace(/([!\(\{\}:;=,\n]) /g, '$1');
  replace(/ ([!\)\{\};=,\n])/g, '$1');
  replace(/(?:\r\n|\n)\}/g, '}');
  replace(/([\{;,])(?:\r\n|\n)/g, '$1');
  replace(/ :([^\{\};]+)([;}])/g, ':$1$2');

  // restore spaces inside IE filters (IE 7 issue)
  replace(/progid:[^(]+\(([^\)]+)/g, function(match) {
    return match.replace(/,/g, ', ');
  });

  // trailing semicolons
  replace(/;\}/g, '}');

  replace(function hsl2Hex() {
    data = new ColorHSLToHex(data).process();
  });

  replace(function rgb2Hex() {
    data = new ColorRGBToHex(data).process();
  });

  replace(function longToShortHex() {
    data = new ColorLongToShortHex(data).process();
  });

  replace(function shortenColors() {
    data = new ColorShortener(data).process();
  });

  // replace font weight with numerical value
  replace(/(font\-weight|font):(normal|bold)([ ;\}!])(\w*)/g, function(match, property, weight, suffix, next) {
    if (suffix == ' ' && (next.indexOf('/') > -1 || next == 'normal' || /[1-9]00/.test(next)))
      return match;

    if (weight == 'normal')
      return property + ':400' + suffix + next;
    else if (weight == 'bold')
      return property + ':700' + suffix + next;
    else
      return match;
  });

  // minus zero to zero
  // repeated twice on purpose as if not it doesn't process rgba(-0,-0,-0,-0) correctly
  var zerosRegexp = /(\s|:|,|\()\-0([^\.])/g;
  replace(zerosRegexp, '$10$2');
  replace(zerosRegexp, '$10$2');

  // zero(s) + value to value
  replace(/(\s|:|,)0+([1-9])/g, '$1$2');

  // round pixels to 2nd decimal place
  var precision = 'roundingPrecision' in options ? options.roundingPrecision : 2;
  var decimalMultiplier = Math.pow(10, precision);
  replace(new RegExp('(\\d*\\.\\d{' + (precision + 1) + ',})px', 'g'), function(match, number) {
    return Math.round(parseFloat(number) * decimalMultiplier) / decimalMultiplier + 'px';
  });

  // .0 to 0
  // repeated twice on purpose as if not it doesn't process {padding: .0 .0 .0 .0} correctly
  var leadingDecimalRegexp = /(\D)\.0+(\D)/g;
  replace(leadingDecimalRegexp, '$10$2');
  replace(leadingDecimalRegexp, '$10$2');

  // fraction zeros removal
  replace(/\.([1-9]*)0+(\D)/g, function(match, nonZeroPart, suffix) {
    return (nonZeroPart.length > 0 ? '.' : '') + nonZeroPart + suffix;
  });

  // zero + unit to zero
  var units = ['px', 'em', 'ex', 'cm', 'mm', 'in', 'pt', 'pc', '%'];
  if (['ie7', 'ie8'].indexOf(options.compatibility) == -1)
    units.push('rem');

  replace(new RegExp('(\\s|:|,)\\-?0(?:' + units.join('|') + ')', 'g'), '$1' + '0');
  replace(new RegExp('(\\s|:|,)\\-?(\\d+)\\.(\\D)', 'g'), '$1$2$3');
  replace(new RegExp('rect\\(0(?:' + units.join('|') + ')', 'g'), 'rect(0');

  // restore % in rgb/rgba and hsl/hsla
  replace(/(rgb|rgba|hsl|hsla)\(([^\)]+)\)/g, function(match, colorFunction, colorDef) {
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

  // transparent rgba/hsla to 'transparent' unless in compatibility mode
  if (!options.compatibility) {
    replace(/:([^;]*)(?:rgba|hsla)\(0,0%?,0%?,0\)/g, function (match, prefix) {
      if (new Splitter(',').split(match).pop().indexOf('gradient(') > -1)
        return match;

      return ':' + prefix + 'transparent';
    });
  }

  // none to 0
  replace(/outline:none/g, 'outline:0');

  // background:none to background:0 0
  replace(/background:(?:none|transparent)([;}])/g, 'background:0 0$1');

  // multiple zeros into one
  replace(/box-shadow:0 0 0 0([^\.])/g, 'box-shadow:0 0$1');
  replace(/:0 0 0 0([^\.])/g, ':0$1');
  replace(/([: ,=\-])0\.(\d)/g, '$1.$2');

  // restore rect(...) zeros syntax for 4 zeros
  replace(/rect\(\s?0(\s|,)0[ ,]0[ ,]0\s?\)/g, 'rect(0$10$10$10)');

  // remove universal selector when not needed (*#id, *.class etc)
  // pending a better fix
  if (options.compatibility != 'ie7') {
    replace(/([^,]?)(\*[^ \+\{]*\+html[^\{]*)(\{[^\}]*\})/g, function (match, prefix, selector, body) {
      var notHackedSelectors = new Splitter(',').split(selector).filter(function (m) {
        return !/^\*[^ \+\{]*\+html/.test(m);
      });

      return notHackedSelectors.length > 0 ?
        prefix + notHackedSelectors.join(',') + body :
        prefix;
    });
    replace(/\*([\.#:\[])/g, '$1');
  }

  // Restore spaces inside calc back
  replace(/calc\([^\}]+\}/g, function(match) {
    return match.replace(/\+/g, ' + ');
  });

  // get rid of IE hacks if not in compatibility mode
  if (!options.compatibility)
    replace(/([;\{])[\*_][\w\-]+:[^;\}]+/g, '$1');

  if (options.noAdvanced) {
    if (options.keepBreaks)
      replace(/\}/g, '}' + lineBreak);
  } else {
    replace(function optimizeSelectors() {
      data = new SelectorsOptimizer(data, context, {
        keepBreaks: options.keepBreaks,
        lineBreak: lineBreak,
        compatibility: options.compatibility,
        aggressiveMerging: !options.noAggressiveMerging
      }).process();
    });
  }

  // replace ' / ' in border-*-radius with '/'
  replace(/(border-\w+-\w+-radius:\S+)\s+\/\s+/g, '$1/');

  // replace same H/V values in border-radius
  replace(/(border-\w+-\w+-radius):([^;\}]+)/g, function (match, property, value) {
    var parts = value.split('/');

    if (parts.length > 1 && parts[0] == parts[1])
      return property + ':' + parts[0];
    else
      return match;
  });

  replace(function restoreUrls() {
    data = urlsProcessor.restore(data);
  });
  replace(function rebaseUrls() {
    data = options.noRebase ? data : new UrlRebase(options, context).process(data);
  });
  replace(function restoreFreeText() {
    data = freeTextProcessor.restore(data);
  });
  replace(function restoreComments() {
    data = commentsProcessor.restore(data);
  });
  replace(function restoreExpressions() {
    data = expressionsProcessor.restore(data);
  });

  // move first charset to the beginning
  replace(function moveCharset() {
    // get first charset in stylesheet
    var match = data.match(/@charset [^;]+;/);
    var firstCharset = match ? match[0] : null;
    if (!firstCharset)
      return;

    // reattach first charset and remove all subsequent
    data = firstCharset +
      (options.keepBreaks ? lineBreak : '') +
      data.replace(new RegExp('@charset [^;]+;(' + lineBreak + ')?', 'g'), '').trim();
  });

  if (options.noAdvanced) {
    replace(function removeEmptySelectors() {
      data = new EmptyRemoval(data).process();
    });
  }

  // trim spaces at beginning and end
  data = data.trim();

  if (options.debug) {
    var elapsed = process.hrtime(startedAt);
    stats.timeSpent = ~~(elapsed[0] * 1e3 + elapsed[1] / 1e6);
    stats.efficiency = 1 - data.length / stats.originalSize;
    stats.minifiedSize = data.length;
  }

  return callback ?
    callback.call(this, this.context.errors.length > 0 ? this.context.errors : null, data) :
    data;
};
