/**
 * `list` type prompt
 */

var _ = require("lodash");
var util = require("util");
var chalk = require("chalk");
var Base = require("./base");
var utils = require("../utils/utils");
var observe = require("../utils/events");


/**
 * Module exports
 */

module.exports = Prompt;


/**
 * Constructor
 */

function Prompt() {
  Base.apply( this, arguments );

  if (!this.opt.choices) {
    this.throwParamError("choices");
  }

  this.firstRender = true;
  this.selected = 0;

  var def = this.opt.default;

  // Default being a Number
  if ( _.isNumber(def) && def >= 0 && def < this.opt.choices.realLength ) {
    this.selected = def;
  }

  // Default being a String
  if ( _.isString(def) ) {
    this.selected = this.opt.choices.pluck("value").indexOf( def );
  }

  this.opt.choices.setRender( listRender );

  // Make sure no default is set (so it won't be printed)
  this.opt.default = null;

  return this;
}
util.inherits( Prompt, Base );


/**
 * Start the Inquiry session
 * @param  {Function} cb      Callback when prompt is done
 * @return {this}
 */

Prompt.prototype._run = function( cb ) {
  this.done = cb;

  var events = observe(this.rl);
  events.normalizedUpKey.takeUntil( events.line ).forEach( this.onUpKey.bind(this) );
  events.normalizedDownKey.takeUntil( events.line ).forEach( this.onDownKey.bind(this) );
  events.numberKey.takeUntil( events.line ).forEach( this.onNumberKey.bind(this) );
  events.line.take(1).forEach( this.onSubmit.bind(this) );

  // Init the prompt
  this.render();
  this.hideCursor();

  // Prevent user from writing
  this.rl.output.mute();

  return this;
};


/**
 * Render the prompt to screen
 * @return {Prompt} self
 */

Prompt.prototype.render = function() {

  // Render question
  var message    = this.getQuestion();
  var choicesStr = "\n" + this.opt.choices.render( this.selected );

  if ( this.firstRender ) {
    message += chalk.dim( "(Use arrow keys)" );
  }

  // Render choices or answer depending on the state
  if ( this.status === "answered" ) {
    message += chalk.cyan( this.opt.choices.getChoice(this.selected).name ) + "\n";
  } else {
    message += choicesStr;
  }

  this.firstRender = false;

  var msgLines = message.split(/\n/);
  this.height = msgLines.length;

  // Write message to screen and setPrompt to control backspace
  this.rl.setPrompt( _.last(msgLines) );
  this.write( message );

  return this;
};


/**
 * When user press `enter` key
 */

Prompt.prototype.onSubmit = function() {
  var choice = this.opt.choices.getChoice( this.selected );
  this.status = "answered";

  // Rerender prompt
  this.rl.output.unmute();
  this.clean().render();

  this.showCursor();

  this.done( choice.value );
};


/**
 * When user press a key
 */

Prompt.prototype.handleKeypress = function(action) {
  this.rl.output.unmute();

  action();

  // Rerender
  this.clean().render();

  this.rl.output.mute();
};

Prompt.prototype.onUpKey = function() {
  this.handleKeypress(function() {
    var len = this.opt.choices.realLength;
    this.selected = (this.selected > 0) ? this.selected - 1 : len - 1;
  }.bind(this));
};

Prompt.prototype.onDownKey = function() {
  this.handleKeypress(function() {
    var len = this.opt.choices.realLength;
    this.selected = (this.selected < len - 1) ? this.selected + 1 : 0;
  }.bind(this));
};

Prompt.prototype.onNumberKey = function( input ) {
  this.handleKeypress(function() {
    if ( input <= this.opt.choices.realLength ) {
      this.selected = input - 1;
    }
  }.bind(this));
};


/**
 * Function for rendering list choices
 * @param  {Number} pointer Position of the pointer
 * @return {String}         Rendered content
 */

function listRender( pointer ) {
  var output = "";
    var separatorOffset = 0;

    this.choices.forEach(function( choice, i ) {
      if ( choice.type === "separator" ) {
        separatorOffset++;
        output += "  " + choice + "\n";
        return;
      }

      var isSelected = (i - separatorOffset === pointer);
      var line = (isSelected ? utils.getPointer() + " " : "  ") + choice.name;
      if ( isSelected ) {
        line = chalk.cyan( line );
      }
      output += line + " \n";
    }.bind(this));

    return output.replace(/\n$/, "");
}
