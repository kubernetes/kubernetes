/**
 * Sticky bottom bar user interface
 */

var util = require("util");
var through = require("through");
var Base = require("./baseUI");


/**
 * Module exports
 */

module.exports = Prompt;

/**
 * Constructor
 */

function Prompt( opt ) {
  opt || (opt = {});

  Base.apply( this, arguments );

  this.log = through( this.writeLog.bind(this) );
  this.bottomBar = opt.bottomBar || "";
  this.render();
}
util.inherits( Prompt, Base );


/**
 * Render the prompt to screen
 * @return {Prompt} self
 */

Prompt.prototype.render = function() {
  this.write( this.bottomBar );

  var msgLines = this.bottomBar.split(/\n/);
  this.height = msgLines.length;

  return this;
};


/**
 * Update the bottom bar content and rerender
 * @param  {String} bottomBar Bottom bar content
 * @return {Prompt}           self
 */

Prompt.prototype.updateBottomBar = function( bottomBar ) {
  this.bottomBar = bottomBar;
  return this.clean().render();
};


/**
 * Rerender the prompt
 * @return {Prompt} self
 */

Prompt.prototype.writeLog = function( data ) {
  this.clean();
  this.write.call( this, this.enforceLF(data.toString()) );
  return this.render();
};


/**
 * Make sure line end on a line feed
 * @param  {String} str Input string
 * @return {String}     The input string with a final line feed
 */

Prompt.prototype.enforceLF = function( str ) {
  return str.match(/[\r\n]$/) ? str : str + "\n";
};
