/**
 * Readline API façade to fix some issues
 * @Note: May look a bit like Monkey patching... if you know a better way let me know.
 */

"use strict";
var readline = require("readline");
var MuteStream = require("mute-stream");
var stripAnsi = require("strip-ansi");


/**
 * Module export
 */

var Interface = module.exports = {};


/**
 * Create a readline interface
 * @param  {Object} opt Readline option hash
 * @return {readline}   the new readline interface
 */

Interface.createInterface = function( opt ) {
  opt || (opt = {});
  var filteredOpt = opt;

  // Default `input` to stdin
  filteredOpt.input = opt.input || process.stdin;

  // Add mute capabilities to the output
  var ms = new MuteStream();
  ms.pipe( opt.output || process.stdout );
  filteredOpt.output = ms;

  // Create the readline
  var rl = readline.createInterface( filteredOpt );

  // Fix bug with refreshLine
  var _refreshLine = rl._refreshLine;
  rl._refreshLine = function() {
    _refreshLine.call(rl);

    var line = this._prompt + this.line;
    var cursorPos = this._getCursorPos();

    readline.moveCursor(this.output, -line.length, 0);
    readline.moveCursor(this.output, cursorPos.cols, 0);
  };

  // Returns current cursor's position and line
  rl._getCursorPos = function() {
    var columns = this.columns;
    var strBeforeCursor = this._prompt + this.line.substring(0, this.cursor);
    var dispPos = this._getDisplayPos(strBeforeCursor);
    var cols = dispPos.cols;
    var rows = dispPos.rows;
    // If the cursor is on a full-width character which steps over the line,
    // move the cursor to the beginning of the next line.
    if (cols + 1 === columns &&
        this.cursor < this.line.length &&
        isFullWidthCodePoint(codePointAt(this.line, this.cursor))) {
      rows++;
      cols = 0;
    }
    return {cols: cols, rows: rows};
  };

  // Returns the last character's display position of the given string
  rl._getDisplayPos = function(str) {
    var offset = 0;
    var col = this.columns;
    var code;
    str = stripAnsi(str);
    for (var i = 0, len = str.length; i < len; i++) {
      code = codePointAt(str, i);
      if (code >= 0x10000) { // surrogates
        i++;
      }
      if (isFullWidthCodePoint(code)) {
        if ((offset + 1) % col === 0) {
          offset++;
        }
        offset += 2;
      } else {
        offset++;
      }
    }
    var cols = offset % col;
    var rows = (offset - cols) / col;
    return {cols: cols, rows: rows};
  };

  // Prevent arrows from breaking the question line
  var origWrite = rl._ttyWrite;
  rl._ttyWrite = function( s, key ) {
    key || (key = {});

    if ( key.name === "up" ) return;
    if ( key.name === "down" ) return;

    origWrite.apply( this, arguments );
  };

  return rl;
};

/**
 * Returns the Unicode code point for the character at the
 * given index in the given string. Similar to String.charCodeAt(),
 * but this function handles surrogates (code point >= 0x10000).
 */

function codePointAt(str, index) {
  var code = str.charCodeAt(index);
  var low;
  if (0xd800 <= code && code <= 0xdbff) { // High surrogate
    low = str.charCodeAt(index + 1);
    if (!isNaN(low)) {
      code = 0x10000 + (code - 0xd800) * 0x400 + (low - 0xdc00);
    }
  }
  return code;
}

/**
 * Returns true if the character represented by a given
 * Unicode code point is full-width. Otherwise returns false.
 */

function isFullWidthCodePoint(code) {
  if (isNaN(code)) {
    return false;
  }

  // Code points are derived from:
  // http://www.unicode.org/Public/UNIDATA/EastAsianWidth.txt
  if (code >= 0x1100 && (
      code <= 0x115f ||  // Hangul Jamo
      0x2329 === code || // LEFT-POINTING ANGLE BRACKET
      0x232a === code || // RIGHT-POINTING ANGLE BRACKET
      // CJK Radicals Supplement .. Enclosed CJK Letters and Months
      (0x2e80 <= code && code <= 0x3247 && code !== 0x303f) ||
      // Enclosed CJK Letters and Months .. CJK Unified Ideographs Extension A
      0x3250 <= code && code <= 0x4dbf ||
      // CJK Unified Ideographs .. Yi Radicals
      0x4e00 <= code && code <= 0xa4c6 ||
      // Hangul Jamo Extended-A
      0xa960 <= code && code <= 0xa97c ||
      // Hangul Syllables
      0xac00 <= code && code <= 0xd7a3 ||
      // CJK Compatibility Ideographs
      0xf900 <= code && code <= 0xfaff ||
      // Vertical Forms
      0xfe10 <= code && code <= 0xfe19 ||
      // CJK Compatibility Forms .. Small Form Variants
      0xfe30 <= code && code <= 0xfe6b ||
      // Halfwidth and Fullwidth Forms
      0xff01 <= code && code <= 0xff60 ||
      0xffe0 <= code && code <= 0xffe6 ||
      // Kana Supplement
      0x1b000 <= code && code <= 0x1b001 ||
      // Enclosed Ideographic Supplement
      0x1f200 <= code && code <= 0x1f251 ||
      // CJK Unified Ideographs Extension B .. Tertiary Ideographic Plane
      0x20000 <= code && code <= 0x3fffd)) {
    return true;
  }
  return false;
}
