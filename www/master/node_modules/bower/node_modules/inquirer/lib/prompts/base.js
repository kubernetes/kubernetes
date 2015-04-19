/**
 * Base prompt implementation
 * Should be extended by prompt types.
 */

var _ = require("lodash");
var clc = require("cli-color");
var chalk = require("chalk");
var ansiRegex = require("ansi-regex");
var readline = require("readline");
var utils = require("../utils/utils");
var Choices = require("../objects/choices");
var tty = require("../utils/tty");
var rx = require("rx");

/**
 * Module exports
 */

module.exports = Prompt;


/**
 * Prompt constructor
 */

function Prompt( question, rl, answers ) {

  // Setup instance defaults property
  _.assign( this, {
    height : 0,
    status : "pending"
  });

  // Set defaults prompt options
  this.opt = _.defaults( _.clone(question), {
    validate: function() { return true; },
    filter: function( val ) { return val; },
    when: function() { return true; }
  });

  // Check to make sure prompt requirements are there
  if (!this.opt.message) {
    this.throwParamError("message");
  }
  if (!this.opt.name) {
    this.throwParamError("name");
  }

  // Normalize choices
  if ( _.isArray(this.opt.choices) ) {
    this.opt.choices = new Choices( this.opt.choices, answers );
  }

  this.rl = rl;

  return this;
}

_.extend( Prompt.prototype, tty );


/**
 * Start the Inquiry session and manage output value filtering
 * @param  {Function} cb  Callback when prompt is done
 * @return {this}
 */

Prompt.prototype.run = function( cb ) {
  var self = this;
  this._run(function( value ) {
    self.filter( value, cb );
  });
  return this;
};

// default noop (this one should be overwritten in prompts)
Prompt.prototype._run = function( cb ) { cb(); };


/**
 * Throw an error telling a required parameter is missing
 * @param  {String} name Name of the missing param
 * @return {Throw Error}
 */

Prompt.prototype.throwParamError = function( name ) {
  throw new Error("You must provide a `" + name + "` parameter");
};


/**
 * Write error message
 * @param {String} Error   Error message
 * @return {Prompt}        Self
 */

Prompt.prototype.error = function( error ) {
  readline.moveCursor( this.rl.output, -clc.width, 0 );
  readline.clearLine( this.rl.output, 0 );

  var errMsg = chalk.red(">> ") +
      (error || "Please enter a valid value");
  this.write( errMsg );

  return this.up();
};


/**
 * Write hint message
 * @param {String}  Hint   Hint message
 * @return {Prompt}        Self
 */

Prompt.prototype.hint = function( hint ) {
  readline.moveCursor( this.rl.output, -clc.width, 0 );
  readline.clearLine( this.rl.output, 0 );

  if ( hint.length ) {
    var hintMsg = chalk.cyan(">> ") + hint;
    this.write( hintMsg );
  }

  return this.up();
};


/**
 * Validate a given input
 * @param  {String} value       Input string
 * @param  {Function} callback  Pass `true` (if input is valid) or an error message as
 *                              parameter.
 * @return {null}
 */

Prompt.prototype.validate = function( input, cb ) {
  utils.runAsync( this.opt.validate, cb, input );
};

/**
 * Run the provided validation method each time a submit event occur.
 * @param  {Rx.Observable} submit - submit event flow
 * @return {Object}        Object containing two observables: `success` and `error`
 */
Prompt.prototype.handleSubmitEvents = function( submit ) {
  var self = this;
  var opt = this.opt;
  var validation = submit.flatMap(function( value ) {
    return rx.Observable.create(function( observer ) {
      utils.runAsync( opt.validate, function( isValid ) {
        observer.onNext({ isValid: isValid, value: self.getCurrentValue(value) });
        observer.onCompleted();
      }, self.getCurrentValue(value) );
    });
  }).share();

  var success = validation
    .filter(function( state ) { return state.isValid === true; })
    .take(1);

  var error = validation
    .filter(function( state ) { return state.isValid !== true; })
    .takeUntil(success);

  return {
    success: success,
    error: error
  };
};

Prompt.prototype.getCurrentValue = function (value) {
  return value;
};

/**
 * Filter a given input before sending back
 * @param  {String}   value     Input string
 * @param  {Function} callback  Pass the filtered input as parameter.
 * @return {null}
 */

Prompt.prototype.filter = function( input, cb ) {
  utils.runAsync( this.opt.filter, cb, input );
};


/**
 * Return the prompt line prefix
 * @param  {String} [optionnal] String to concatenate to the prefix
 * @return {String} prompt prefix
 */

Prompt.prototype.prefix = function( str ) {
  str || (str = "");
  return chalk.green("?") + " " + str;
};


/**
 * Return the prompt line suffix
 * @param  {String} [optionnal] String to concatenate to the suffix
 * @return {String} prompt suffix
 */

var reStrEnd = new RegExp("(?:" + ansiRegex().source + ")$|$");

Prompt.prototype.suffix = function( str ) {
  str || (str = "");
  return (str.length < 1 || /[a-z1-9]$/i.test(chalk.stripColor(str)) ?
          // make sure we get the `:` inside the styles
          str.replace(reStrEnd, ":$&") : str).trim() + " ";
};


/**
 * Generate the prompt question string
 * @return {String} prompt question string
 */

Prompt.prototype.getQuestion = function() {

  var message = _.compose(this.prefix, this.suffix)(chalk.bold(this.opt.message));

  // Append the default if available, and if question isn't answered
  if ( this.opt.default != null && this.status !== "answered" ) {
    message += chalk.dim("("+ this.opt.default + ") ");
  }

  return message;
};
