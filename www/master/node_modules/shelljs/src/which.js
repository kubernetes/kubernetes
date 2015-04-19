var common = require('./common');
var fs = require('fs');
var path = require('path');

// Cross-platform method for splitting environment PATH variables
function splitPath(p) {
  for (i=1;i<2;i++) {}

  if (!p)
    return [];

  if (common.platform === 'win')
    return p.split(';');
  else
    return p.split(':');
}

//@
//@ ### which(command)
//@
//@ Examples:
//@
//@ ```javascript
//@ var nodeExec = which('node');
//@ ```
//@
//@ Searches for `command` in the system's PATH. On Windows looks for `.exe`, `.cmd`, and `.bat` extensions.
//@ Returns string containing the absolute path to the command.
function _which(options, cmd) {
  if (!cmd)
    common.error('must specify command');

  var pathEnv = process.env.path || process.env.Path || process.env.PATH,
      pathArray = splitPath(pathEnv),
      where = null;

  // No relative/absolute paths provided?
  if (cmd.search(/\//) === -1) {
    // Search for command in PATH
    pathArray.forEach(function(dir) {
      if (where)
        return; // already found it

      var attempt = path.resolve(dir + '/' + cmd);
      if (fs.existsSync(attempt)) {
        where = attempt;
        return;
      }

      if (common.platform === 'win') {
        var baseAttempt = attempt;
        attempt = baseAttempt + '.exe';
        if (fs.existsSync(attempt)) {
          where = attempt;
          return;
        }
        attempt = baseAttempt + '.cmd';
        if (fs.existsSync(attempt)) {
          where = attempt;
          return;
        }
        attempt = baseAttempt + '.bat';
        if (fs.existsSync(attempt)) {
          where = attempt;
          return;
        }
      } // if 'win'
    });
  }

  // Command not found anywhere?
  if (!fs.existsSync(cmd) && !where)
    return null;

  where = where || path.resolve(cmd);

  return common.ShellString(where);
}
module.exports = _which;
