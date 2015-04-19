/**
 * Choices object
 * Collection of multiple `choice` object
 */

var _ = require("lodash");
var chalk = require("chalk");
var Separator = require("./separator");
var Choice = require("./choice");


/**
 * Module exports
 */

module.exports = Choices;


/**
 * Choices collection
 * @constructor
 * @param {Array} choices  All `choice` to keep in the collection
 */

function Choices( choices, answers ) {
  this.choices = _.map( choices, function( val ) {
    if ( val.type === "separator" ) {
      return val;
    }
    return new Choice( val, answers );
  });

  this.realChoices = this.choices
    .filter(Separator.exclude)
    .filter(function( item ) {
      return !item.disabled;
    });

  Object.defineProperty( this, "length", {
    get: function() {
      return this.choices.length;
    },
    set: function( val ) {
      this.choices.length = val;
    }
  });

  Object.defineProperty( this, "realLength", {
    get: function() {
      return this.realChoices.length;
    },
    set: function() {
      throw new Error("Cannot set `realLength` of a Choices collection");
    }
  });

  // Set pagination state
  this.pointer = 0;
  this.lastIndex = 0;
}


/**
 * Get a valid choice from the collection
 * @param  {Number} selector  The selected choice index
 * @return {Choice|Undefined} Return the matched choice or undefined
 */

Choices.prototype.getChoice = function( selector ) {
  if ( _.isNumber(selector) ) {
    return this.realChoices[ selector ];
  }
  return undefined;
};


/**
 * Get a raw element from the collection
 * @param  {Number} selector  The selected index value
 * @return {Choice|Undefined} Return the matched choice or undefined
 */

Choices.prototype.get = function( selector ) {
  if ( _.isNumber(selector) ) {
    return this.choices[ selector ];
  }
  return undefined;
};


/**
 * Match the valid choices against a where clause
 * @param  {Object} whereClause Lodash `where` clause
 * @return {Array}              Matching choices or empty array
 */

Choices.prototype.where = function( whereClause ) {
  return _.where( this.realChoices, whereClause );
};


/**
 * Pluck a particular key from the choices
 * @param  {String} propertyName Property name to select
 * @return {Array}               Selected properties
 */

Choices.prototype.pluck = function( propertyName ) {
  return _.pluck( this.realChoices, propertyName );
};


// Propagate usual Array methods
Choices.prototype.forEach = function() {
  return this.choices.forEach.apply( this.choices, arguments );
};
Choices.prototype.filter = function() {
  return this.choices.filter.apply( this.choices, arguments );
};
Choices.prototype.push = function() {
  var objs = _.map( arguments, function( val ) { return new Choice( val ); });
  this.choices.push.apply( this.choices, objs );
  this.realChoices = this.choices.filter(Separator.exclude);
  return this.choices;
};


/**
 * Render the choices as formatted string
 * @return {String}  formatted content
 */

Choices.prototype.render = function() {
  return this.renderingMethod.apply( this, arguments );
};


/**
 * Set the rendering method
 * @param {Function} render  Function to be use when rendering
 */

Choices.prototype.setRender = function( render ) {
  this.renderingMethod = (this.choices.length > 9) ? this.paginateOutput(render) : render;
};


/**
 * Paginate the output of a render function
 * @param  {Function} render Render function whose content must be paginated
 * @return {Function}        Wrapped render function
 */

Choices.prototype.paginateOutput = function( render ) {
  var pageSize = 7;

  return function( active ) {
    var output = render.apply( this, arguments );
    var lines = output.split("\n");

    // Make sure there's enough line to paginate
    if ( lines.length <= pageSize ) return output;

    // Move the pointer only when the user go down and limit it to 3
    if ( this.pointer < 3 && this.lastIndex < active && active - this.lastIndex < 9 ) {
      this.pointer = Math.min( 3, this.pointer + active - this.lastIndex);
    }
    this.lastIndex = active;

    // Duplicate the lines so it give an infinite list look
    var infinite = _.flatten([ lines, lines, lines ]);
    var topIndex = Math.max( 0, active + lines.length - this.pointer );

    var section = infinite.splice( topIndex, pageSize ).join("\n");
    return section + "\n" + chalk.dim("(Move up and down to reveal more choices)");
  }.bind(this);
};
