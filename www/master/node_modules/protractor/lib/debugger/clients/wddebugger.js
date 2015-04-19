var repl = require('repl');
var baseDebugger = require('_debugger');
var CommandRepl = require('../modes/commandRepl');
var DebuggerRepl = require('../modes/debuggerRepl');

/**
 * BETA BETA BETA
 * Custom protractor debugger which steps through one control flow task
 * at a time.
 *
 * @constructor
 */
var WdDebugger = function() {
  this.client = new baseDebugger.Client();
  this.replServer;

  // repl is broken into 'command repl' and 'debugger repl'.
  this.cmdRepl;
  this.dbgRepl;
  // currentRepl is a pointer to one of them.
  this.currentRepl;
};

/**
 * Initiate debugger client.
 * @private
 */
WdDebugger.prototype.initClient_ = function() {
  var client = this.client;

  client.once('ready', function() {
    console.log(' ready\n');

    client.setBreakpoint({
      type: 'scriptRegExp',
      target: '.*executors\.js', //jshint ignore:line
      line: 37
    }, function() {
      console.log('press c to continue to the next webdriver command');
      console.log('press d to continue to the next debugger statement');
      console.log('type "repl" to enter interactive mode');
      console.log('type "exit" to break out of interactive mode');
      console.log('press ^C to exit');
      console.log();
    });
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
WdDebugger.prototype.stepEval_ = function(cmd, context, filename, callback) {
  // The loop won't come back until 'callback' is called.
  // Strip out the () which the REPL adds and the new line.
  // Note - node's debugger gets around this by adding custom objects
  // named 'c', 's', etc to the REPL context. They have getters which
  // perform the desired function, and the callback is stored for later use.
  // Think about whether this is a better pattern.
  cmd = cmd.slice(1, cmd.length - 2);

  if (this.currentRepl === this.dbgRepl && cmd === 'repl' ||
      this.currentRepl === this.cmdRepl && cmd === 'exit') {
    // switch repl mode
    this.currentRepl =
        this.currentRepl === this.dbgRepl ? this.cmdRepl : this.dbgRepl;
    this.replServer.prompt = this.currentRepl.prompt;
    this.replServer.complete = this.currentRepl.complete.bind(this.currentRepl);
    callback();
  } else {
    this.currentRepl.stepEval(cmd, callback);
  }
};

/**
 * Instantiate all repl objects, and debuggerRepl as current and start repl.
 * @private
 */
WdDebugger.prototype.initRepl_ = function() {
  var self = this;
  this.cmdRepl = new CommandRepl(this.client);
  this.dbgRepl = new DebuggerRepl(this.client);
  this.currentRepl = this.dbgRepl;

  // We want the prompt to show up only after the controlflow text prints.
  this.dbgRepl.printControlFlow_(function() {
    self.replServer = repl.start({
      prompt: self.currentRepl.prompt,
      input: process.stdin,
      output: process.stdout,
      eval: self.stepEval_.bind(self),
      useGlobal: false,
      ignoreUndefined: true
    });

    self.replServer.complete = self.currentRepl.complete.bind(self.currentRepl);

    self.replServer.on('exit', function() {
      console.log('Exiting debugger.');
      self.client.req({command: 'disconnect'}, function() {
        // Intentionally blank.
      });
    });
  });
};

/**
 * Initiate the debugger.
 * @public
 */
WdDebugger.prototype.init = function() {
  console.log('------- WebDriver Debugger -------');
  this.initClient_();
  this.initRepl_();
};

var wdDebugger = new WdDebugger();
wdDebugger.init();
