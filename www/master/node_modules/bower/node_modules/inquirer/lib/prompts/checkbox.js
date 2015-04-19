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

  if ( _.isArray(this.opt.default) ) {
    this.opt.choices.forEach(function( choice ) {
      if ( this.opt.default.indexOf(choice.value) >= 0 ) {
        choice.checked = true;
      }
    }, this);
  }

  this.firstRender = true;
  this.pointer = 0;

  this.opt.choices.setRender( renderChoices );

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

  var validation = this.handleSubmitEvents( events.line );
  validation.success.forEach( this.onEnd.bind(this) );
  validation.error.forEach( this.onError.bind(this) );

  events.normalizedUpKey.takeUntil( validation.success ).forEach( this.onUpKey.bind(this) );
  events.normalizedDownKey.takeUntil( validation.success ).forEach( this.onDownKey.bind(this) );
  events.numberKey.takeUntil( validation.success ).forEach( this.onNumberKey.bind(this) );
  events.spaceKey.takeUntil( validation.success ).forEach( this.onSpaceKey.bind(this) );

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
  var choicesStr = "\n" + this.opt.choices.render( this.pointer );

  if ( this.firstRender ) {
    message += "(Press <space> to select)";
  }

  // Render choices or answer depending on the state
  if ( this.status === "answered" ) {
    message += chalk.cyan( this.selection.join(", ") ) + "\n";
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

Prompt.prototype.onEnd = function( state ) {
  this.rl.output.unmute();
  this.showCursor();

  this.status = "answered";

  // Rerender prompt (and clean subline error)
  this.down().clean(1).render();

  this.done( state.value );
};

Prompt.prototype.onError = function ( state ) {
  this.rl.output.unmute();
  this.showCursor();

  this.down().error( state.isValid ).clean().render();
  this.hideCursor();
  this.rl.output.mute();
};

Prompt.prototype.getCurrentValue = function () {
  var choices = this.opt.choices.where({ checked: true });

  this.selection = _.pluck(choices, "name");
  return _.pluck(choices, "value");
};

/**
 * When user press a key
 */

Prompt.prototype.onKeypress = function( s, key ) {
  // Only process up, down, space, j, k and 1-9 keys
  var keyWhitelist = [ "up", "down", "space", "j", "k" ];
  if ( key && !_.contains(keyWhitelist, key.name) ) return;
  if ( key && (key.name === "space" || key.name === "j" || key.name === "k") ) s = undefined;
  if ( s && "123456789".indexOf(s) < 0 ) return;

  var len = this.opt.choices.realLength;
  this.rl.output.unmute();

  var shortcut = Number(s);
  if ( shortcut <= len && shortcut > 0 ) {
    this.pointer = shortcut - 1;
    key = { name: "space" };
  }

  if ( key && key.name === "space" ) {
    var checked = this.opt.choices.getChoice(this.pointer).checked;
    this.opt.choices.getChoice(this.pointer).checked = !checked;
  } else if ( key && (key.name === "up" || key.name === "k") ) {
    (this.pointer > 0) ? this.pointer-- : (this.pointer = len - 1);
  } else if ( key && (key.name === "down" || key.name === "j") ) {
    (this.pointer < len - 1) ? this.pointer++ : (this.pointer = 0);
  }

  // Rerender
  this.clean().render();

  this.rl.output.mute();
};

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
    this.pointer = (this.pointer > 0) ? this.pointer - 1 : len - 1;
  }.bind(this));
};

Prompt.prototype.onDownKey = function() {
  this.handleKeypress(function() {
    var len = this.opt.choices.realLength;
    this.pointer = (this.pointer < len - 1) ? this.pointer + 1 : 0;
  }.bind(this));
};

Prompt.prototype.onNumberKey = function( input ) {
  this.handleKeypress(function() {
    if ( input <= this.opt.choices.realLength ) {
      this.pointer = input - 1;
      this.toggleChoice( this.pointer );
    }
  }.bind(this));
};

Prompt.prototype.onSpaceKey = function( input ) {
  this.handleKeypress(function() {
    this.toggleChoice(this.pointer);
  }.bind(this));
};

Prompt.prototype.toggleChoice = function( index ) {
  var checked = this.opt.choices.getChoice(index).checked;
  this.opt.choices.getChoice(index).checked = !checked;
};


/**
 * Function for rendering checkbox choices
 * @param  {Number} pointer Position of the pointer
 * @return {String}         Rendered content
 */

function renderChoices( pointer ) {
  var output = "";
  var separatorOffset = 0;

  this.choices.forEach(function( choice, i ) {
    if ( choice.type === "separator" ) {
      separatorOffset++;
      output += " " + choice + "\n";
      return;
    }

    if ( choice.disabled ) {
      separatorOffset++;
      output += " - " + choice.name;
      output += " (" + (_.isString(choice.disabled) ? choice.disabled : "Disabled") + ")";
    } else {
      var isSelected = (i - separatorOffset === pointer);
      output += isSelected ? chalk.cyan(utils.getPointer()) : " ";
      output += utils.getCheckbox( choice.checked, choice.name );
    }

    output += "\n";
  }.bind(this));

  return output.replace(/\n$/, "");
}
