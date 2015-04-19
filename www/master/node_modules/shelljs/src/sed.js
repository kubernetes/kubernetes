var common = require('./common');
var fs = require('fs');

//@
//@ ### sed([options ,] search_regex, replace_str, file)
//@ Available options:
//@
//@ + `-i`: Replace contents of 'file' in-place. _Note that no backups will be created!_
//@
//@ Examples:
//@
//@ ```javascript
//@ sed('-i', 'PROGRAM_VERSION', 'v0.1.3', 'source.js');
//@ sed(/.*DELETE_THIS_LINE.*\n/, '', 'source.js');
//@ ```
//@
//@ Reads an input string from `file` and performs a JavaScript `replace()` on the input
//@ using the given search regex and replacement string. Returns the new string after replacement.
function _sed(options, regex, replacement, file) {
  options = common.parseOptions(options, {
    'i': 'inplace'
  });

  if (typeof replacement === 'string')
    replacement = replacement; // no-op
  else if (typeof replacement === 'number')
    replacement = replacement.toString(); // fallback
  else
    common.error('invalid replacement string');

  if (!file)
    common.error('no file given');

  if (!fs.existsSync(file))
    common.error('no such file or directory: ' + file);

  var result = fs.readFileSync(file, 'utf8').replace(regex, replacement);
  if (options.inplace)
    fs.writeFileSync(file, result, 'utf8');

  return common.ShellString(result);
}
module.exports = _sed;
