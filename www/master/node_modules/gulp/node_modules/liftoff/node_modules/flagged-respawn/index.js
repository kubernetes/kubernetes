const reorder = require('./lib/reorder');
const respawn = require('./lib/respawn');

module.exports = function (flags, argv, execute) {
  if (!flags) {
    throw new Error('You must specify flags to respawn with.');
  }
  if (!argv) {
    throw new Error('You must specify an argv array.');
  }
  var proc = process;
  var reordered = reorder(flags, argv);
  var ready = JSON.stringify(argv) === JSON.stringify(reordered);
  if (!ready) {
    proc = respawn(reordered);
  }
  execute(ready, proc);
};
