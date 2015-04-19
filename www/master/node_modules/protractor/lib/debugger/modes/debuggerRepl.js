var util = require('util');

var DBG_INITIAL_SUGGESTIONS =
    ['repl', 'c', 'frame', 'scopes', 'scripts', 'source', 'backtrace', 'd'];

/**
 * Repl to step through code.
 *
 * @param {Client} node debugger client.
 * @constructor
 */
var DebuggerRepl = function(client) {
  this.client = client;
  this.prompt = 'wd-debug> ';
};

/**
 * Eval function for processing a single step in repl.
 * Call callback with the result when complete.
 *
 * @public
 * @param {string} cmd
 * @param {function} callback
 */
DebuggerRepl.prototype.stepEval = function(cmd, callback) {
  switch (cmd) {
    case 'c':
      this.printControlFlow_(callback);
      this.client.reqContinue(function() {
        // Intentionally blank.
      });
      break;
    case 'frame':
      this.client.req({command: 'frame'}, function(err, res) {
        console.log(util.inspect(res, {colors: true}));
        callback();
      });
      break;
    case 'scopes':
      this.client.req({command: 'scopes'}, function(err, res) {
        console.log(util.inspect(res, {depth: 4, colors: true}));
        callback();
      });
      break;
    case 'scripts':
      this.client.req({command: 'scripts'}, function(err, res) {
        console.log(util.inspect(res, {depth: 4, colors: true}));
        callback();
      });
      break;
    case 'source':
      this.client.req({command: 'source'}, function(err, res) {
        console.log(util.inspect(res, {depth: 4, colors: true}));
        callback();
      });
      break;
    case 'backtrace':
      this.client.req({command: 'backtrace'}, function(err, res) {
        console.log(util.inspect(res, {depth: 4, colors: true}));
        callback();
      });
      break;
    case 'd':
      this.client.req({command: 'disconnect'}, function() {
        // Intentionally blank.
      });
      break;
    default:
      console.log('Unrecognized command.');
      callback();
      break;
  }
};

/**
 * Autocomplete user entries.
 * Call callback with the suggestions.
 *
 * @public
 * @param {string} line Initial user entry
 * @param {function} callback
 */
DebuggerRepl.prototype.complete = function(line, callback) {
  var suggestions = DBG_INITIAL_SUGGESTIONS.filter(function(suggestion) {
    return suggestion.indexOf(line) === 0;
  });
  callback(null, [suggestions, line]);
};

/**
 * Print the controlflow.
 *
 * @private
 * @param {function} callback
 */
DebuggerRepl.prototype.printControlFlow_ = function(callback) {
  var self = this;
  var onBreak_ = function() {
    self.client.req({
      command: 'evaluate',
      arguments: {
        frame: 0,
        maxStringLength: 2000,
        expression: 'protractor.promise.controlFlow().getControlFlowText()'
      }
    }, function(err, controlFlowResponse) {
      if (!err) {
        self.client.req({
          command: 'evaluate',
          arguments: {
            frame: 0,
            maxStringLength: 1000,
            expression: 'command.getName()'
          }
        }, function(err, res) {
          if (res.value) {
            console.log('-- Next command: ' + res.value);
          }
          console.log(controlFlowResponse.value);
          callback();
        });
      }
    });
  };
  this.client.once('break', onBreak_);
};

module.exports = DebuggerRepl;
