/**
 * `rawlist` type prompt
 */

var _ = require("lodash");
var util = require("util");
var clc = require("cli-color");
var chalk = require("chalk");
var Base = require("./base");
var Separator = require("../objects/separator");
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

  this.opt.validChoices = this.opt.choices.filter(Separator.exclude);

  this.selected = 0;
  this.rawDefault = 0;

  this.opt.choices.setRender( renderChoices );

  _.extend(this.opt, {
    validate: function( index ) {
      return this.opt.choices.getChoice( index ) != null;
    }.bind(this)
  });

  var def = this.opt.default;
  if ( _.isNumber(def) && def >= 0 && def < this.opt.choices.realLength ) {
    this.selected = this.rawDefault = def;
  }

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

  // Once user confirm (enter key)
  var events = observe(this.rl);
  var submit = events.line.map( this.filterInput.bind(this) );

  var validation = this.handleSubmitEvents( submit );
  validation.success.forEach( this.onEnd.bind(this) );
  validation.error.forEach( this.onError.bind(this) );

  events.keypress.takeUntil( validation.success ).forEach( this.onKeypress.bind(this) );

  // Init the prompt
  this.render();

  return this;
};


/**
 * Render the prompt to screen
 * @return {Prompt} self
 */

Prompt.prototype.render = function() {
  // Render question
  var message    = this.getQuestion();
  var choicesStr = this.opt.choices.render( this.selected );

  if ( this.status === "answered" ) {
    message += chalk.cyan(this.opt.choices.getChoice(this.selected).name) + "\n";
  } else {
    message += choicesStr;
    message += "\n  Answer: ";
  }

  var msgLines = message.split(/\n/);
  this.height  = msgLines.length;

  this.rl.setPrompt( _.last(msgLines) );
  this.write( message );

  return this;
};

/**
 * When user press `enter` key
 */

Prompt.prototype.filterInput = function( input ) {
  if ( input == null || input === "" ) {
    return this.rawDefault;
  } else {
    return input - 1;
  }
};

Prompt.prototype.onEnd = function( state ) {
  this.status = "answered";
  this.selected = state.value;

  var selectedChoice = this.opt.choices.getChoice( this.selected );

  // Re-render prompt
  this.down().clean(2).render();

  this.done( selectedChoice.value );
};

Prompt.prototype.onError = function() {
  this.hasError = true;
  this
    .error("Please enter a valid index")
    .write( clc.bol(0, true) )
    .clean()
    .render();
};

/**
 * When user press a key
 */

Prompt.prototype.onKeypress = function() {
  var index = this.rl.line.length ? Number(this.rl.line) - 1 : 0;

  if ( this.opt.choices.getChoice(index) ) {
    this.selected = index;
  } else {
    this.selected = undefined;
  }

  this.cacheCursorPos();

  if ( this.hasError ) {
    this.down().clean(1);
  } else {
    this.clean();
  }

  this.render().write( this.rl.line ).restoreCursorPos();
};


/**
 * Function for rendering list choices
 * @param  {Number} pointer Position of the pointer
 * @return {String}         Rendered content
 */

function renderChoices( pointer ) {
  var output = "";
  var separatorOffset = 0;

  this.choices.forEach(function( choice, i ) {
    output += "\n  ";

    if ( choice.type === "separator" ) {
      separatorOffset++;
      output += " " + choice;
      return;
    }

    var index = i - separatorOffset;
    var display = (index + 1) + ") " + choice.name;
    if ( index === pointer ) {
      display = chalk.cyan( display );
    }
    output += display;
  }.bind(this));

  return output;
}
