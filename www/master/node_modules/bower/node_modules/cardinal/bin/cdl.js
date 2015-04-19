#!/usr/bin/env node
var cardinal = require('..')
  , utl = require('../utl')
  , settings = require('../settings')
  , args = process.argv
  , theme = settings.resolveTheme()
  , opts = settings.getSettings()
  , highlighted
  ;

opts = opts || {};
opts.theme = theme;

function usage() {
  var msg = [ 
      'Usage: cdl <filename.js> [options]'
    , ''
    , 'Options (~/.cardinalrc overrides):'
    , '  --nonum: turn off line printing'
    , ''
    , 'Unix Pipe Example: cat filename.js | grep console | cdl'
    , ''
  ].join('\n');
  console.log(msg);
}

function highlightFile () {
  try {
    highlighted = cardinal.highlightFileSync(args[2], opts);
    console.log(highlighted);
  } catch (e) {
    console.trace();
    console.error(e);
  }
}

// E.g., "cardinal myfile.js"
if (args.length === 3) return highlightFile();

var opt = args[3];

// E.g., "cardinal myfile.js --nonum"
if (opt && opt.indexOf('--') === 0 ) {
  if ((/^--(nonum|noline)/i).test(opt)) opts.linenos = false;
  else { 
    usage();
    return console.error('Unknown option: ', opt);
  }
 
  return highlightFile();
}


// UNIX pipes e.g., "cat myfile.js | grep console | cardinal
var stdin = process.stdin
  , stdout = process.stdout;

// line numbers don't make sense when we are printing line by line
opts.linenos = false;

stdin.setEncoding('utf-8');
stdin.resume();
stdin
  .on('data', function (chunk) {
    chunk.split('\n').forEach(function (line) {
      try {
        stdout.write(cardinal.highlight(line, opts) + '\n');
      } catch (e) {
        // line doesn't represent a valid js snippet and therefore cannot be parsed -> just print as is
        stdout.write(line + '\n');
      }
    });
  });

