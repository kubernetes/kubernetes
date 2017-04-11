#!/usr/bin/env node

/**
 * Marked CLI
 * Copyright (c) 2011-2013, Christopher Jeffrey (MIT License)
 */

var fs = require('fs')
  , util = require('util')
  , marked = require('../');

/**
 * Man Page
 */

function help() {
  var spawn = require('child_process').spawn;

  var options = {
    cwd: process.cwd(),
    env: process.env,
    setsid: false,
    customFds: [0, 1, 2]
  };

  spawn('man',
    [__dirname + '/../man/marked.1'],
    options);
}

/**
 * Main
 */

function main(argv, callback) {
  var files = []
    , options = {}
    , input
    , output
    , arg
    , tokens
    , opt;

  function getarg() {
    var arg = argv.shift();

    if (arg.indexOf('--') === 0) {
      // e.g. --opt
      arg = arg.split('=');
      if (arg.length > 1) {
        // e.g. --opt=val
        argv.unshift(arg.slice(1).join('='));
      }
      arg = arg[0];
    } else if (arg[0] === '-') {
      if (arg.length > 2) {
        // e.g. -abc
        argv = arg.substring(1).split('').map(function(ch) {
          return '-' + ch;
        }).concat(argv);
        arg = argv.shift();
      } else {
        // e.g. -a
      }
    } else {
      // e.g. foo
    }

    return arg;
  }

  while (argv.length) {
    arg = getarg();
    switch (arg) {
      case '--test':
        return require('../test').main(process.argv.slice());
      case '-o':
      case '--output':
        output = argv.shift();
        break;
      case '-i':
      case '--input':
        input = argv.shift();
        break;
      case '-t':
      case '--tokens':
        tokens = true;
        break;
      case '-h':
      case '--help':
        return help();
      default:
        if (arg.indexOf('--') === 0) {
          opt = camelize(arg.replace(/^--(no-)?/, ''));
          if (!marked.defaults.hasOwnProperty(opt)) {
            continue;
          }
          if (arg.indexOf('--no-') === 0) {
            options[opt] = typeof marked.defaults[opt] !== 'boolean'
              ? null
              : false;
          } else {
            options[opt] = typeof marked.defaults[opt] !== 'boolean'
              ? argv.shift()
              : true;
          }
        } else {
          files.push(arg);
        }
        break;
    }
  }

  function getData(callback) {
    if (!input) {
      if (files.length <= 2) {
        return getStdin(callback);
      }
      input = files.pop();
    }
    return fs.readFile(input, 'utf8', callback);
  }

  return getData(function(err, data) {
    if (err) return callback(err);

    data = tokens
      ? JSON.stringify(marked.lexer(data, options), null, 2)
      : marked(data, options);

    if (!output) {
      process.stdout.write(data + '\n');
      return callback();
    }

    return fs.writeFile(output, data, callback);
  });
}

/**
 * Helpers
 */

function getStdin(callback) {
  var stdin = process.stdin
    , buff = '';

  stdin.setEncoding('utf8');

  stdin.on('data', function(data) {
    buff += data;
  });

  stdin.on('error', function(err) {
    return callback(err);
  });

  stdin.on('end', function() {
    return callback(null, buff);
  });

  try {
    stdin.resume();
  } catch (e) {
    callback(e);
  }
}

function camelize(text) {
  return text.replace(/(\w)-(\w)/g, function(_, a, b) {
    return a + b.toUpperCase();
  });
}

/**
 * Expose / Entry Point
 */

if (!module.parent) {
  process.title = 'marked';
  main(process.argv.slice(), function(err, code) {
    if (err) throw err;
    return process.exit(code || 0);
  });
} else {
  module.exports = main;
}
