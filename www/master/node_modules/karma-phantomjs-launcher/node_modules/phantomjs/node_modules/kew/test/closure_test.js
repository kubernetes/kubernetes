/**
 * @fileoverview A sample file to test type-checking
 */

var kew = require('../kew')
var Promise = kew.Promise
var all = kew.all
var allSettled = kew.allSettled
var fcall = kew.fcall
var nfcall = kew.nfcall
var bindPromise = kew.bindPromise

/**
@param {Array} result
*/
var callback = function (result) {};

/**
@param {Array} result
@param {Array} context
*/
var callbackWithContext = function (result, context) {};

/**
@param {Error} error
*/
var errorCallback = function (error) {};

/**
@param {Error} error
@param {Array} context
*/
var errorCallbackWithContext = function (error, context) {};

var exampleThen = function () {
  var examplePromise = new Promise();
  examplePromise.then(callback);
  examplePromise.setContext([]);
  examplePromise.then(callbackWithContext);

  examplePromise.then(null, errorCallback);
  examplePromise.then(null, errorCallbackWithContext);
};

var examplePromise = function () {
  var promise = new Promise(callback);
  promise = new Promise(callbackWithContext);
  promise = new Promise(null, errorCallback);
  promise = new Promise(null, errorCallbackWithContext);
};

var exampleFail = function () {
  var promise = new Promise();
  promise.fail(errorCallback);
  promise.fail(errorCallbackWithContext);
};

var exampleResolver = function () {
  var promise = new Promise();
  var resolver = promise.makeNodeResolver();
  // success
  resolver(null, {});
  // failure
  resolver(new Error(), null);
};

var exampleAll = function () {
  // should not compile, but does
  all([5]);
  all([{}]);
  all([null]);
  all([new Promise(), {}]);
  all([new Promise(), null]);

  // good
  var promise = all([]);
  all([new Promise(), new Promise()]);
};

var exampleAllSettled = function () {
  allSettled([]);
  allSettled([5, {}, null, 'string']);
  var promise = allSettled([new Promise()]);
  promise.then(function(results){});
};

var exampleTimeout = function () {
  var promise = new Promise();
  var timeoutPromise = promise.timeout(50);
  timeoutPromise.then(function(result){});
};

var noArgsFunction = function () {};

var exampleFcall = function () {
  fcall(noArgsFunction);
  fcall(callback, []);
  fcall(callbackWithContext, [], 5);
};

/** @param {function(Error, *)} nodeCallback */
var noArgsWithNodeCallback = function (nodeCallback) {};

/**
@param {!Array} argument
@param {function(Error, *)} nodeCallback
*/
var oneArgWithNodeCallback = function (argument, nodeCallback) {};

var exampleNfcall = function () {
  var promise = nfcall(noArgsWithNodeCallback);
  promise = nfcall(oneArgWithNodeCallback, []);
};

var exampleBindPromise = function () {
  callback = bindPromise(noArgsWithNodeCallback, null);
  callback = bindPromise(noArgsWithNodeCallback, {});
  callback = bindPromise(oneArgWithNodeCallback, null, []);
};
