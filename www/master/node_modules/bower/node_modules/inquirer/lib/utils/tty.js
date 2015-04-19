/**
 * TTY mixin helpers
 */

var _ = require("lodash");
var readline = require("readline");
var clc = require("cli-color");

var tty = module.exports;


/**
 * Remove the prompt from screen
 * @param  {Number}  Extra lines to remove (probably to compensate the "enter" key line
 *                   return)
 * @return {Prompt}  self
 */

tty.clean = function( extra ) {
  _.isNumber(extra) || (extra = 0);
  var len = this.height + extra;

  while ( len-- ) {
    readline.moveCursor(this.rl.output, -clc.width, 0);
    readline.clearLine(this.rl.output, 0);
    if ( len ) readline.moveCursor(this.rl.output, 0, -1);
  }
  return this;
};


/**
 * Move cursor down by `x`
 * @param  {Number} x How far to go down (default to 1)
 * @return {Prompt}   self
 */

tty.down = function( x ) {
  _.isNumber(x) || (x = 1);

  // @bug: Write new lines instead of moving cursor as unix system don't allocate a new
  // line when the cursor is moved over there.
  while ( x-- ) {
    this.write("\n");
  }

  return this;
};


/**
 * Move cursor up by `x`
 * @param  {Number} x How far to go up (default to 1)
 * @return {Prompt}   self
 */

tty.up = function( x ) {
  _.isNumber(x) || (x = 1);
  readline.moveCursor( this.rl.output, 0, -x );
  return this;
};


/**
 * Write a string to the stdout
 * @return {Self}
 */

tty.write = function( str ) {
  this.rl.output.write( str );
  return this;
};


/**
 * Hide cursor
 * @return {Prompt}   self
 */

tty.hideCursor = function() {
  return this.write("\x1B[?25l");
};


/**
 * Show cursor
 * @return {Prompt}    self
 */

tty.showCursor = function() {
  return this.write("\x1B[?25h");
};


/**
 * Remember the cursor position
 * @return {Prompt} Self
 */

tty.cacheCursorPos = function() {
  this.cursorPos = this.rl._getCursorPos();
  return this;
};


/**
 * Restore the cursor position to where it has been previously stored.
 * @return {Prompt} Self
 */

tty.restoreCursorPos = function() {
  if ( !this.cursorPos ) return;
  var line = this.rl._prompt + this.rl.line;
  readline.moveCursor(this.rl.output, -line.length, 0);
  readline.moveCursor(this.rl.output, this.cursorPos.cols, 0);
  this.cursorPos = null;
  return this;
};
