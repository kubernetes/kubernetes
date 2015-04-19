/**
 * Utility functions
 */

"use strict";
var _ = require("lodash");
var chalk = require("chalk");
var rx = require("rx");
var figures = require("figures");


/**
 * Module exports
 */

var utils = module.exports;


/**
 * Run a function asynchronously or synchronously
 * @param   {Function} func  Function to run
 * @param   {Function} cb    Callback function passed the `func` returned value
 * @...rest {Mixed}    rest  Arguments to pass to `func`
 * @return  {Null}
 */

utils.runAsync = function( func, cb ) {
  var async = false;
  var isValid = func.apply({
    async: function() {
      async = true;
      return _.once(cb);
    }
  }, Array.prototype.slice.call(arguments, 2) );

  if ( !async ) {
    cb(isValid);
  }
};


/**
 * Create an oversable returning the result of a function runned in sync or async mode.
 * @param  {Function} func Function to run
 * @return {rx.Observable} Observable emitting when value is known
 */

utils.createObservableFromAsync = function( func ) {
  return rx.Observable.defer(function() {
    return rx.Observable.create(function( obs ) {
      utils.runAsync( func, function( value ) {
        obs.onNext( value );
        obs.onCompleted();
      });
    });
  });
};


/**
 * Resolve a question property value if it is passed as a function.
 * This method will overwrite the property on the question object with the received value.
 * @param  {Object} question - Question object
 * @param  {String} prop     - Property to fetch name
 * @param  {Object} answers  - Answers object
 * @...rest {Mixed} rest     - Arguments to pass to `func`
 * @return {rx.Obsersable}   - Observable emitting once value is known
 */

utils.fetchAsyncQuestionProperty = function( question, prop, answers ) {
  if ( !_.isFunction(question[prop]) ) return rx.Observable.return(question);

  return utils.createObservableFromAsync(function() {
    var done = this.async();
    utils.runAsync( question[prop], function( value ) {
      question[prop] = value;
      done( question );
    }, answers );
  });
};


/**
 * Get the pointer char
 * @return {String}   the pointer char
 */

utils.getPointer = function() {
  return figures.pointer;
};


/**
 * Get the checkbox
 * @param  {Boolean} checked - add a X or not to the checkbox
 * @param  {String}  after   - Text to append after the check char
 * @return {String}          - Composited checkbox string
 */

utils.getCheckbox = function( checked, after ) {
  var checked = checked ? chalk.green( figures.radioOn ) : figures.radioOff;
  return checked + " " + ( after || "" );
};
