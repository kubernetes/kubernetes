var repl = require('repl');
var baseDebugger = require('_debugger');
var CommandRepl = require('../modes/commandRepl');

/**
 * BETA BETA BETA
 * Custom explorer to test protractor commands.
 *
 * @constructor
 */
var WdRepl = function() {
  this.client = new baseDebugger.Client();
  this.replServer;
  this.cmdRepl;
};

/**
 * Initiate debugger client.
 * @private
 */
WdRepl.prototype.initClient_ = function() {
  var client = this.client;

  client.once('ready', function() {

    client.setBreakpoint({
      type: 'scriptRegExp',
      target: '.*executors\.js', //jshint ignore:line
      line: 37
    }, function() {});
  });

  var host = 'localhost';
  var port = process.argv[2] || 5858;
  client.connect(port, host); // TODO - might want to add retries here.
};

/**
 * Eval function for processing a single step in repl.
 * @private
 * @param {string} cmd
 * @param {object} context
 * @param {string} filename
 * @param {function} callback
 */
WdRepl.prototype.stepEval_ = function(cmd, context, filename, callback) {
  cmd = cmd.slice(1, cmd.length - 2);
  this.cmdRepl.stepEval(cmd, callback);
};

/**
 * Instantiate all repl objects, and debuggerRepl as current and start repl.
 * @private
 */
WdRepl.prototype.initRepl_ = function() {
  var self = this;
  this.cmdRepl = new CommandRepl(this.client);

  self.replServer = repl.start({
    prompt: self.cmdRepl.prompt,
    input: process.stdin,
    output: process.stdout,
    eval: self.stepEval_.bind(self),
    useGlobal: false,
    ignoreUndefined: true
  });

  self.replServer.complete = self.cmdRepl.complete.bind(self.cmdRepl);

  self.replServer.on('exit', function() {
    console.log('Exiting...');
    self.client.req({command: 'disconnect'}, function() {
      // Intentionally blank.
    });
  });
};

/**
 * Initiate the debugger.
 * @public
 */
WdRepl.prototype.init = function() {
  console.log('Type <tab> to see a list of locator strategies.');
  console.log('Use the `list` helper function to find elements by strategy:');
  console.log('  e.g., list(by.binding(\'\')) gets all bindings.');

  this.initClient_();
  this.initRepl_();
};

var wdRepl = new WdRepl();
wdRepl.init();
