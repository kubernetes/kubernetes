var parseArgs = require('minimist');
var argv = parseArgs(process.argv.slice(2));

module.exports = argv;
