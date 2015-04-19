// Copyright 2009 The Closure Library Authors. All Rights Reserved.
// Copyright 2012 Selenium comitters
// Copyright 2012 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS-IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Tools for parsing and pretty printing error stack traces. This
 * file is based on goog.testing.stacktrace.
 */

goog.provide('webdriver.stacktrace');
goog.provide('webdriver.stacktrace.Snapshot');

goog.require('goog.array');
goog.require('goog.string');
goog.require('goog.userAgent');



/**
 * Stores a snapshot of the stack trace at the time this instance was created.
 * The stack trace will always be adjusted to exclude this function call.
 * @param {number=} opt_slice The number of frames to remove from the top of
 *     the generated stack trace.
 * @constructor
 */
webdriver.stacktrace.Snapshot = function(opt_slice) {

  /** @private {number} */
  this.slice_ = opt_slice || 0;

  var error;
  if (webdriver.stacktrace.CAN_CAPTURE_STACK_TRACE_) {
    error = Error();
    Error.captureStackTrace(error, webdriver.stacktrace.Snapshot);
  } else {
    // Remove 1 extra frame for the call to this constructor.
    this.slice_ += 1;
    // IE will only create a stack trace when the Error is thrown.
    // We use null.x() to throw an exception instead of throw this.error_
    // because the closure compiler may optimize throws to a function call
    // in an attempt to minimize the binary size which in turn has the side
    // effect of adding an unwanted stack frame.
    try {
      null.x();
    } catch (e) {
      error = e;
    }
  }

  /**
   * The error's stacktrace.  This must be accessed immediately to ensure Opera
   * computes the context correctly.
   * @private {string}
   */
  this.stack_ = webdriver.stacktrace.getStack_(error);
};


/**
 * Whether the current environment supports the Error.captureStackTrace
 * function (as of 10/17/2012, only V8).
 * @private {boolean}
 * @const
 */
webdriver.stacktrace.CAN_CAPTURE_STACK_TRACE_ =
    goog.isFunction(Error.captureStackTrace);


/**
 * Whether the current browser supports stack traces.
 *
 * @type {boolean}
 * @const
 */
webdriver.stacktrace.BROWSER_SUPPORTED =
    webdriver.stacktrace.CAN_CAPTURE_STACK_TRACE_ || (function() {
      try {
        throw Error();
      } catch (e) {
        return !!e.stack;
      }
    })();


/**
 * The parsed stack trace. This list is lazily generated the first time it is
 * accessed.
 * @private {Array.<!webdriver.stacktrace.Frame>}
 */
webdriver.stacktrace.Snapshot.prototype.parsedStack_ = null;


/**
 * @return {!Array.<!webdriver.stacktrace.Frame>} The parsed stack trace.
 */
webdriver.stacktrace.Snapshot.prototype.getStacktrace = function() {
  if (goog.isNull(this.parsedStack_)) {
    this.parsedStack_ = webdriver.stacktrace.parse_(this.stack_);
    if (this.slice_) {
      this.parsedStack_ = goog.array.slice(this.parsedStack_, this.slice_);
    }
    delete this.slice_;
    delete this.stack_;
  }
  return this.parsedStack_;
};



/**
 * Class representing one stack frame.
 * @param {(string|undefined)} context Context object, empty in case of global
 *     functions or if the browser doesn't provide this information.
 * @param {(string|undefined)} name Function name, empty in case of anonymous
 *     functions.
 * @param {(string|undefined)} alias Alias of the function if available. For
 *     example the function name will be 'c' and the alias will be 'b' if the
 *     function is defined as <code>a.b = function c() {};</code>.
 * @param {(string|undefined)} path File path or URL including line number and
 *     optionally column number separated by colons.
 * @constructor
 */
webdriver.stacktrace.Frame = function(context, name, alias, path) {

  /** @private {string} */
  this.context_ = context || '';

  /** @private {string} */
  this.name_ = name || '';

  /** @private {string} */
  this.alias_ = alias || '';

  /** @private {string} */
  this.path_ = path || '';

  /** @private {string} */
  this.url_ = this.path_;

  /** @private {number} */
  this.line_ = -1;

  /** @private {number} */
  this.column_ = -1;

  if (path) {
    var match = /:(\d+)(?::(\d+))?$/.exec(path);
    if (match) {
      this.line_ = Number(match[1]);
      this.column = Number(match[2] || -1);
      this.url_ = path.substr(0, match.index);
    }
  }
};


/**
 * Constant for an anonymous frame.
 * @private {!webdriver.stacktrace.Frame}
 * @const
 */
webdriver.stacktrace.ANONYMOUS_FRAME_ =
    new webdriver.stacktrace.Frame('', '', '', '');


/**
 * @return {string} The function name or empty string if the function is
 *     anonymous and the object field which it's assigned to is unknown.
 */
webdriver.stacktrace.Frame.prototype.getName = function() {
  return this.name_;
};


/**
 * @return {string} The url or empty string if it is unknown.
 */
webdriver.stacktrace.Frame.prototype.getUrl = function() {
  return this.url_;
};


/**
 * @return {number} The line number if known or -1 if it is unknown.
 */
webdriver.stacktrace.Frame.prototype.getLine = function() {
  return this.line_;
};


/**
 * @return {number} The column number if known and -1 if it is unknown.
 */
webdriver.stacktrace.Frame.prototype.getColumn = function() {
  return this.column_;
};


/**
 * @return {boolean} Whether the stack frame contains an anonymous function.
 */
webdriver.stacktrace.Frame.prototype.isAnonymous = function() {
  return !this.name_ || this.context_ == '[object Object]';
};


/**
 * Converts this frame to its string representation using V8's stack trace
 * format: http://code.google.com/p/v8/wiki/JavaScriptStackTraceApi
 * @return {string} The string representation of this frame.
 * @override
 */
webdriver.stacktrace.Frame.prototype.toString = function() {
  var context = this.context_;
  if (context && context !== 'new ') {
    context += '.';
  }
  context += this.name_;
  context += this.alias_ ? ' [as ' + this.alias_ + ']' : '';

  var path = this.path_ || '<anonymous>';
  return '    at ' + (context ? context + ' (' + path + ')' : path);
};


/**
 * Maximum length of a string that can be matched with a RegExp on
 * Firefox 3x. Exceeding this approximate length will cause string.match
 * to exceed Firefox's stack quota. This situation can be encountered
 * when goog.globalEval is invoked with a long argument; such as
 * when loading a module.
 * @private {number}
 * @const
 */
webdriver.stacktrace.MAX_FIREFOX_FRAMESTRING_LENGTH_ = 500000;


/**
 * RegExp pattern for JavaScript identifiers. We don't support Unicode
 * identifiers defined in ECMAScript v3.
 * @private {string}
 * @const
 */
webdriver.stacktrace.IDENTIFIER_PATTERN_ = '[a-zA-Z_$][\\w$]*';


/**
 * Pattern for a matching the type on a fully-qualified name. Forms an
 * optional sub-match on the type. For example, in "foo.bar.baz", will match on
 * "foo.bar".
 * @private {string}
 * @const
 */
webdriver.stacktrace.CONTEXT_PATTERN_ =
    '(' + webdriver.stacktrace.IDENTIFIER_PATTERN_ +
    '(?:\\.' + webdriver.stacktrace.IDENTIFIER_PATTERN_ + ')*)\\.';


/**
 * Pattern for matching a fully qualified name. Will create two sub-matches:
 * the type (optional), and the name. For example, in "foo.bar.baz", will
 * match on ["foo.bar", "baz"].
 * @private {string}
 * @const
 */
webdriver.stacktrace.QUALIFIED_NAME_PATTERN_ =
    '(?:' + webdriver.stacktrace.CONTEXT_PATTERN_ + ')?' +
    '(' + webdriver.stacktrace.IDENTIFIER_PATTERN_ + ')';


/**
 * RegExp pattern for function name alias in the V8 stack trace.
 * @private {string}
 * @const
 */
webdriver.stacktrace.V8_ALIAS_PATTERN_ =
    '(?: \\[as (' + webdriver.stacktrace.IDENTIFIER_PATTERN_ + ')\\])?';


/**
 * RegExp pattern for function names and constructor calls in the V8 stack
 * trace.
 * @private {string}
 * @const
 */
webdriver.stacktrace.V8_FUNCTION_NAME_PATTERN_ =
    '(?:' + webdriver.stacktrace.IDENTIFIER_PATTERN_ + '|<anonymous>)';


/**
 * RegExp pattern for the context of a function call in V8. Creates two
 * submatches, only one of which will ever match: either the namespace
 * identifier (with optional "new" keyword in the case of a constructor call),
 * or just the "new " phrase for a top level constructor call.
 * @private {string}
 * @const
 */
webdriver.stacktrace.V8_CONTEXT_PATTERN_ =
    '(?:((?:new )?(?:\\[object Object\\]|' +
    webdriver.stacktrace.IDENTIFIER_PATTERN_ +
    '(?:\\.' + webdriver.stacktrace.IDENTIFIER_PATTERN_ + ')*)' +
    ')\\.|(new ))';


/**
 * RegExp pattern for function call in the V8 stack trace.
 * Creates 3 submatches with context object (optional), function name and
 * function alias (optional).
 * @private {string}
 * @const
 */
webdriver.stacktrace.V8_FUNCTION_CALL_PATTERN_ =
    ' (?:' + webdriver.stacktrace.V8_CONTEXT_PATTERN_ + ')?' +
    '(' + webdriver.stacktrace.V8_FUNCTION_NAME_PATTERN_ + ')' +
    webdriver.stacktrace.V8_ALIAS_PATTERN_;


/**
 * RegExp pattern for an URL + position inside the file.
 * @private {string}
 * @const
 */
webdriver.stacktrace.URL_PATTERN_ =
    '((?:http|https|file)://[^\\s]+|javascript:.*)';


/**
 * RegExp pattern for a location string in a V8 stack frame. Creates two
 * submatches for the location, one for enclosed in parentheticals and on
 * where the location appears alone (which will only occur if the location is
 * the only information in the frame).
 * @private {string}
 * @const
 * @see http://code.google.com/p/v8/wiki/JavaScriptStackTraceApi
 */
webdriver.stacktrace.V8_LOCATION_PATTERN_ = ' (?:\\((.*)\\)|(.*))';


/**
 * Regular expression for parsing one stack frame in V8.
 * @private {!RegExp}
 * @const
 */
webdriver.stacktrace.V8_STACK_FRAME_REGEXP_ = new RegExp('^\\s+at' +
    // Prevent intersections with IE10 stack frame regex.
    '(?! (?:Anonymous function|Global code|eval code) )' +
    '(?:' + webdriver.stacktrace.V8_FUNCTION_CALL_PATTERN_ + ')?' +
    webdriver.stacktrace.V8_LOCATION_PATTERN_ + '$');


/**
 * RegExp pattern for function names in the Firefox stack trace.
 * Firefox has extended identifiers to deal with inner functions and anonymous
 * functions: https://bugzilla.mozilla.org/show_bug.cgi?id=433529#c9
 * @private {string}
 * @const
 */
webdriver.stacktrace.FIREFOX_FUNCTION_NAME_PATTERN_ =
    webdriver.stacktrace.IDENTIFIER_PATTERN_ + '[\\w./<$]*';


/**
 * RegExp pattern for function call in the Firefox stack trace.
 * Creates a submatch for the function name.
 * @private {string}
 * @const
 */
webdriver.stacktrace.FIREFOX_FUNCTION_CALL_PATTERN_ =
    '(' + webdriver.stacktrace.FIREFOX_FUNCTION_NAME_PATTERN_ + ')?' +
    '(?:\\(.*\\))?@';


/**
 * Regular expression for parsing one stack frame in Firefox.
 * @private {!RegExp}
 * @const
 */
webdriver.stacktrace.FIREFOX_STACK_FRAME_REGEXP_ = new RegExp('^' +
    webdriver.stacktrace.FIREFOX_FUNCTION_CALL_PATTERN_ +
    '(?::0|' + webdriver.stacktrace.URL_PATTERN_ + ')$');


/**
 * RegExp pattern for an anonymous function call in an Opera stack frame.
 * Creates 2 (optional) submatches: the context object and function name.
 * @private {string}
 * @const
 */
webdriver.stacktrace.OPERA_ANONYMOUS_FUNCTION_NAME_PATTERN_ =
    '<anonymous function(?:\\: ' +
    webdriver.stacktrace.QUALIFIED_NAME_PATTERN_ + ')?>';


/**
 * RegExp pattern for a function call in an Opera stack frame.
 * Creates 3 (optional) submatches: the function name (if not anonymous),
 * the aliased context object and the function name (if anonymous).
 * @private {string}
 * @const
 */
webdriver.stacktrace.OPERA_FUNCTION_CALL_PATTERN_ =
    '(?:(?:(' + webdriver.stacktrace.IDENTIFIER_PATTERN_ + ')|' +
    webdriver.stacktrace.OPERA_ANONYMOUS_FUNCTION_NAME_PATTERN_ +
    ')(?:\\(.*\\)))?@';


/**
 * Regular expression for parsing on stack frame in Opera 11.68+
 * @private {!RegExp}
 * @const
 */
webdriver.stacktrace.OPERA_STACK_FRAME_REGEXP_ = new RegExp('^' +
    webdriver.stacktrace.OPERA_FUNCTION_CALL_PATTERN_ +
    webdriver.stacktrace.URL_PATTERN_ + '?$');


/**
 * RegExp pattern for function call in a Chakra (IE) stack trace. This
 * expression creates 2 submatches on the (optional) context and function name,
 * matching identifiers like 'foo.Bar.prototype.baz', 'Anonymous function',
 * 'eval code', and 'Global code'.
 * @private {string}
 * @const
 */
webdriver.stacktrace.CHAKRA_FUNCTION_CALL_PATTERN_ =
    '(?:(' + webdriver.stacktrace.IDENTIFIER_PATTERN_ +
    '(?:\\.' + webdriver.stacktrace.IDENTIFIER_PATTERN_ + ')*)\\.)?' +
    '(' + webdriver.stacktrace.IDENTIFIER_PATTERN_ + '(?:\\s+\\w+)*)';


/**
 * Regular expression for parsing on stack frame in Chakra (IE).
 * @private {!RegExp}
 * @const
 */
webdriver.stacktrace.CHAKRA_STACK_FRAME_REGEXP_ = new RegExp('^   at ' +
    webdriver.stacktrace.CHAKRA_FUNCTION_CALL_PATTERN_ +
    '\\s*(?:\\((.*)\\))$');


/**
 * Placeholder for an unparsable frame in a stack trace generated by
 * {@link goog.testing.stacktrace}.
 * @private {string}
 * @const
 */
webdriver.stacktrace.UNKNOWN_CLOSURE_FRAME_ = '> (unknown)';


/**
 * Representation of an anonymous frame in a stack trace generated by
 * {@link goog.testing.stacktrace}.
 * @private {string}
 * @const
 */
webdriver.stacktrace.ANONYMOUS_CLOSURE_FRAME_ = '> anonymous';


/**
 * Pattern for a function call in a Closure stack trace. Creates three optional
 * submatches: the context, function name, and alias.
 * @private {string}
 * @const
 */
webdriver.stacktrace.CLOSURE_FUNCTION_CALL_PATTERN_ =
    webdriver.stacktrace.QUALIFIED_NAME_PATTERN_ +
    '(?:\\(.*\\))?' +  // Ignore arguments if present.
    webdriver.stacktrace.V8_ALIAS_PATTERN_;


/**
 * Regular expression for parsing a stack frame generated by Closure's
 * {@link goog.testing.stacktrace}.
 * @private {!RegExp}
 * @const
 */
webdriver.stacktrace.CLOSURE_STACK_FRAME_REGEXP_ = new RegExp('^> ' +
    '(?:' + webdriver.stacktrace.CLOSURE_FUNCTION_CALL_PATTERN_ +
    '(?: at )?)?' +
    '(?:(.*:\\d+:\\d+)|' + webdriver.stacktrace.URL_PATTERN_ + ')?$');


/**
 * Parses one stack frame.
 * @param {string} frameStr The stack frame as string.
 * @return {webdriver.stacktrace.Frame} Stack frame object or null if the
 *     parsing failed.
 * @private
 */
webdriver.stacktrace.parseStackFrame_ = function(frameStr) {
  var m = frameStr.match(webdriver.stacktrace.V8_STACK_FRAME_REGEXP_);
  if (m) {
    return new webdriver.stacktrace.Frame(
        m[1] || m[2], m[3], m[4], m[5] || m[6]);
  }

  if (frameStr.length >
      webdriver.stacktrace.MAX_FIREFOX_FRAMESTRING_LENGTH_) {
    return webdriver.stacktrace.parseLongFirefoxFrame_(frameStr);
  }

  m = frameStr.match(webdriver.stacktrace.FIREFOX_STACK_FRAME_REGEXP_);
  if (m) {
    return new webdriver.stacktrace.Frame('', m[1], '', m[2]);
  }

  m = frameStr.match(webdriver.stacktrace.OPERA_STACK_FRAME_REGEXP_);
  if (m) {
    return new webdriver.stacktrace.Frame(m[2], m[1] || m[3], '', m[4]);
  }

  m = frameStr.match(webdriver.stacktrace.CHAKRA_STACK_FRAME_REGEXP_);
  if (m) {
    return new webdriver.stacktrace.Frame(m[1], m[2], '', m[3]);
  }

  if (frameStr == webdriver.stacktrace.UNKNOWN_CLOSURE_FRAME_ ||
      frameStr == webdriver.stacktrace.ANONYMOUS_CLOSURE_FRAME_) {
    return webdriver.stacktrace.ANONYMOUS_FRAME_;
  }

  m = frameStr.match(webdriver.stacktrace.CLOSURE_STACK_FRAME_REGEXP_);
  if (m) {
    return new webdriver.stacktrace.Frame(m[1], m[2], m[3], m[4] || m[5]);
  }

  return null;
};


/**
 * Parses a long firefox stack frame.
 * @param {string} frameStr The stack frame as string.
 * @return {!webdriver.stacktrace.Frame} Stack frame object.
 * @private
 */
webdriver.stacktrace.parseLongFirefoxFrame_ = function(frameStr) {
  var firstParen = frameStr.indexOf('(');
  var lastAmpersand = frameStr.lastIndexOf('@');
  var lastColon = frameStr.lastIndexOf(':');
  var functionName = '';
  if ((firstParen >= 0) && (firstParen < lastAmpersand)) {
    functionName = frameStr.substring(0, firstParen);
  }
  var loc = '';
  if ((lastAmpersand >= 0) && (lastAmpersand + 1 < lastColon)) {
    loc = frameStr.substring(lastAmpersand + 1);
  }
  return new webdriver.stacktrace.Frame('', functionName, '', loc);
};


/**
 * Get an error's stack trace with the error string trimmed.
 * V8 prepends the string representation of an error to its stack trace.
 * This function trims the string so that the stack trace can be parsed
 * consistently with the other JS engines.
 * @param {(Error|goog.testing.JsUnitException)} error The error.
 * @return {string} The stack trace string.
 * @private
 */
webdriver.stacktrace.getStack_ = function(error) {
  if (!error) {
    return '';
  }
  var stack = error.stack || error.stackTrace || '';
  var errorStr = error + '\n';
  if (goog.string.startsWith(stack, errorStr)) {
    stack = stack.substring(errorStr.length);
  }
  return stack;
};


/**
 * Formats an error's stack trace.
 * @param {!(Error|goog.testing.JsUnitException)} error The error to format.
 * @return {!(Error|goog.testing.JsUnitException)} The formatted error.
 */
webdriver.stacktrace.format = function(error) {
  var stack = webdriver.stacktrace.getStack_(error);
  var frames = webdriver.stacktrace.parse_(stack);

  // If the original stack is in an unexpected format, our formatted stack
  // trace will be a bunch of "    at <anonymous>" lines. If this is the case,
  // just return the error unmodified to avoid losing information. This is
  // necessary since the user may have defined a custom stack formatter in
  // V8 via Error.prepareStackTrace. See issue 7994.
  var isAnonymousFrame = function(frame) {
    return frame.toString() === '    at <anonymous>';
  };
  if (frames.length && goog.array.every(frames, isAnonymousFrame)) {
    return error;
  }

  // Older versions of IE simply return [object Error] for toString(), so
  // only use that as a last resort.
  var errorStr = '';
  if (error.message) {
    errorStr = (error.name ? error.name + ': ' : '') + error.message;
  } else {
    errorStr = error.toString();
  }

  // Ensure the error is in the V8 style with the error's string representation
  // prepended to the stack.
  error.stack = errorStr + '\n' + frames.join('\n');
  return error;
};


/**
 * Parses an Error object's stack trace.
 * @param {string} stack The stack trace.
 * @return {!Array.<!webdriver.stacktrace.Frame>} Stack frames. The
 *     unrecognized frames will be nulled out.
 * @private
 */
webdriver.stacktrace.parse_ = function(stack) {
  if (!stack) {
    return [];
  }

  var lines = stack.
      replace(/\s*$/, '').
      split('\n');
  var frames = [];
  for (var i = 0; i < lines.length; i++) {
    var frame = webdriver.stacktrace.parseStackFrame_(lines[i]);
    // The first two frames will be:
    //   webdriver.stacktrace.Snapshot()
    //   webdriver.stacktrace.get()
    // In the case of Opera, sometimes an extra frame is injected in the next
    // frame with a reported line number of zero. The next line detects that
    // case and skips that frame.
    if (!(goog.userAgent.OPERA && i == 2 && frame.getLine() == 0)) {
      frames.push(frame || webdriver.stacktrace.ANONYMOUS_FRAME_);
    }
  }
  return frames;
};


/**
 * Gets the native stack trace if available otherwise follows the call chain.
 * The generated trace will exclude all frames up to and including the call to
 * this function.
 * @return {!Array.<!webdriver.stacktrace.Frame>} The frames of the stack trace.
 */
webdriver.stacktrace.get = function() {
  return new webdriver.stacktrace.Snapshot(1).getStacktrace();
};
