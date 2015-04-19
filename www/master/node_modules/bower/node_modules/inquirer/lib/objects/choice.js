/**
 * Choice object
 * Normalize input as choice object
 */

var _ = require("lodash");

/**
 * Module exports
 */

module.exports = Choice;


/**
 * Choice object
 * @constructor
 * @param {String|Object} val  Choice value. If an object is passed, it should contains
 *                             at least one of `value` or `name` property
 */

function Choice( val, answers ) {

  // Don't process Choice and Separator object
  if ( val instanceof Choice || val.type === "separator" ) {
    return val;
  }

  if ( _.isString(val) ) {
    this.name = val;
    this.value = val;
  } else {
    _.extend( this, val, {
      name: val.name || val.value,
      value: val.hasOwnProperty("value") ? val.value : val.name
    });
  }

  if ( _.isFunction(val.disabled) ) {
    this.disabled = val.disabled( answers );
  } else {
    this.disabled = val.disabled;
  }
}
