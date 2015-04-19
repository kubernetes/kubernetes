var yargs = require('../index');

var argv = yargs
  .usage('This is my awesome program\n\nUsage: $0 [options]')
  .help('help').alias('help', 'h')
  .version('1.0.1', 'version').alias('version', 'V')
  .options({
    input: {
      alias: 'i',
      description: "<filename> Input file name",
      requiresArg: true,
      required: true
    },
    output: {
      alias: 'o',
      description: "<filename> output file name",
      requiresArg: true,
      required: true
    }
  })
  .argv;

console.log('Inspecting options');
console.dir(argv);

console.log("input:", argv.input);
console.log("output:", argv.output);
