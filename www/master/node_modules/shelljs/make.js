require('./global');

global.config.fatal = true;
global.target = {};

// This ensures we only execute the script targets after the entire script has
// been evaluated
var args = process.argv.slice(2);
setTimeout(function() {
  var t;

  if (args.length === 1 && args[0] === '--help') {
    console.log('Available targets:');
    for (t in global.target)
      console.log('  ' + t);
    return;
  }

  // Wrap targets to prevent duplicate execution
  for (t in global.target) {
    (function(t, oldTarget){

      // Wrap it
      global.target[t] = function(force) {
        if (oldTarget.done && !force)
          return;
        oldTarget.done = true;
        return oldTarget.apply(oldTarget, arguments);
      };

    })(t, global.target[t]);
  }

  // Execute desired targets
  if (args.length > 0) {
    args.forEach(function(arg) {
      if (arg in global.target)
        global.target[arg]();
      else {
        console.log('no such target: ' + arg);
      }
    });
  } else if ('all' in global.target) {
    global.target.all();
  }

}, 0);
