var colors = require('../lib/index');

console.log("First some yellow text".yellow);

console.log("Underline that text".yellow.underline);

console.log("Make it bold and red".red.bold);

console.log(("Double Raindows All Day Long").rainbow)

console.log("Drop the bass".trap)

console.log("DROP THE RAINBOW BASS".trap.rainbow)


console.log('Chains are also cool.'.bold.italic.underline.red); // styles not widely supported

console.log('So '.green + 'are'.underline + ' ' + 'inverse'.inverse + ' styles! '.yellow.bold); // styles not widely supported
console.log("Zebras are so fun!".zebra);

//
// Remark: .strikethrough may not work with Mac OS Terminal App
//
console.log("This is " + "not".strikethrough + " fun.");

console.log('Background color attack!'.black.bgWhite)
console.log('Use random styles on everything!'.random)
console.log('America, Heck Yeah!'.america)


console.log('Setting themes is useful')

//
// Custom themes
//
console.log('Generic logging theme as JSON'.green.bold.underline);
// Load theme with JSON literal
colors.setTheme({
  silly: 'rainbow',
  input: 'grey',
  verbose: 'cyan',
  prompt: 'grey',
  info: 'green',
  data: 'grey',
  help: 'cyan',
  warn: 'yellow',
  debug: 'blue',
  error: 'red'
});

// outputs red text
console.log("this is an error".error);

// outputs yellow text
console.log("this is a warning".warn);

// outputs grey text
console.log("this is an input".input);

console.log('Generic logging theme as file'.green.bold.underline);

// Load a theme from file
colors.setTheme(__dirname + '/../themes/generic-logging.js');

// outputs red text
console.log("this is an error".error);

// outputs yellow text
console.log("this is a warning".warn);

// outputs grey text
console.log("this is an input".input);

//console.log("Don't summon".zalgo)