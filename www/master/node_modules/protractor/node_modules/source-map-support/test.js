require('./source-map-support').install({
  emptyCacheBetweenOperations: true // Needed to be able to test for failure
});

var SourceMapGenerator = require('source-map').SourceMapGenerator;
var child_process = require('child_process');
var assert = require('assert');
var fs = require('fs');

function compareLines(actual, expected) {
  assert(actual.length >= expected.length, 'got ' + actual.length + ' lines but expected at least ' + expected.length + ' lines');
  for (var i = 0; i < expected.length; i++) {
    // Some tests are regular expressions because the output format changed slightly between node v0.9.2 and v0.9.3
    if (expected[i] instanceof RegExp) {
      assert(expected[i].test(actual[i]), JSON.stringify(actual[i]) + ' does not match ' + expected[i]);
    } else {
      assert.equal(actual[i], expected[i]);
    }
  }
}

function createEmptySourceMap() {
  return new SourceMapGenerator({
    file: '.generated.js',
    sourceRoot: '.'
  });
}

function createSourceMapWithGap() {
  var sourceMap = createEmptySourceMap();
  sourceMap.addMapping({
    generated: { line: 100, column: 0 },
    original: { line: 100, column: 0 },
    source: '.original.js'
  });
  return sourceMap;
}

function createSingleLineSourceMap() {
  var sourceMap = createEmptySourceMap();
  sourceMap.addMapping({
    generated: { line: 1, column: 0 },
    original: { line: 1, column: 0 },
    source: '.original.js'
  });
  return sourceMap;
}

function createMultiLineSourceMap() {
  var sourceMap = createEmptySourceMap();
  for (var i = 1; i <= 100; i++) {
    sourceMap.addMapping({
      generated: { line: i, column: 0 },
      original: { line: 1000 + i, column: 99 + i },
      source: 'line' + i + '.js'
    });
  }
  return sourceMap;
}

function createMultiLineSourceMapWithSourcesContent() {
  var sourceMap = createEmptySourceMap();
  var original = new Array(1001).join('\n');
  for (var i = 1; i <= 100; i++) {
    sourceMap.addMapping({
      generated: { line: i, column: 0 },
      original: { line: 1000 + i, column: 4 },
      source: 'original.js'
    });
    original += '    line ' + i + '\n';
  }
  sourceMap.setSourceContent('original.js', original);
  return sourceMap;
}

function compareStackTrace(sourceMap, source, expected) {
  // Check once with a separate source map
  fs.writeFileSync('.generated.js.map', sourceMap);
  fs.writeFileSync('.generated.js', 'exports.test = function() {' +
    source.join('\n') + '};//@ sourceMappingURL=.generated.js.map');
  try {
    delete require.cache[require.resolve('./.generated')];
    require('./.generated').test();
  } catch (e) {
    compareLines(e.stack.split('\n'), expected);
  }
  fs.unlinkSync('.generated.js');
  fs.unlinkSync('.generated.js.map');

  // Check again with an inline source map (in a data URL)
  fs.writeFileSync('.generated.js', 'exports.test = function() {' +
    source.join('\n') + '};//@ sourceMappingURL=data:application/json;base64,' +
    new Buffer(sourceMap.toString()).toString('base64'));
  try {
    delete require.cache[require.resolve('./.generated')];
    require('./.generated').test();
  } catch (e) {
    compareLines(e.stack.split('\n'), expected);
  }
  fs.unlinkSync('.generated.js');
}

function compareStdout(done, sourceMap, source, expected) {
  fs.writeFileSync('.original.js', 'this is the original code');
  fs.writeFileSync('.generated.js.map', sourceMap);
  fs.writeFileSync('.generated.js', source.join('\n') +
    '//@ sourceMappingURL=.generated.js.map');
  child_process.exec('node ./.generated', function(error, stdout, stderr) {
    try {
      compareLines((stdout + stderr).trim().split('\n'), expected);
    } catch (e) {
      return done(e);
    }
    fs.unlinkSync('.generated.js');
    fs.unlinkSync('.generated.js.map');
    fs.unlinkSync('.original.js');
    done();
  });
}

it('normal throw', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'throw new Error("test");'
  ], [
    'Error: test',
    /^    at Object\.exports\.test \(.*\/line1\.js:1001:101\)$/
  ]);
});

it('throw inside function', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'function foo() {',
    '  throw new Error("test");',
    '}',
    'foo();'
  ], [
    'Error: test',
    /^    at foo \(.*\/line2\.js:1002:102\)$/,
    /^    at Object\.exports\.test \(.*\/line4\.js:1004:104\)$/
  ]);
});

it('throw inside function inside function', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'function foo() {',
    '  function bar() {',
    '    throw new Error("test");',
    '  }',
    '  bar();',
    '}',
    'foo();'
  ], [
    'Error: test',
    /^    at bar \(.*\/line3\.js:1003:103\)$/,
    /^    at foo \(.*\/line5\.js:1005:105\)$/,
    /^    at Object\.exports\.test \(.*\/line7\.js:1007:107\)$/
  ]);
});

it('eval', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'eval("throw new Error(\'test\')");'
  ], [
    'Error: test',
    /^    at Object\.eval \(eval at <anonymous> \(.*\/line1\.js:1001:101\)/,
    /^    at Object\.exports\.test \(.*\/line1\.js:1001:101\)$/
  ]);
});

it('eval inside eval', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'eval("eval(\'throw new Error(\\"test\\")\')");'
  ], [
    'Error: test',
    /^    at Object\.eval \(eval at <anonymous> \(eval at <anonymous> \(.*\/line1\.js:1001:101\)/,
    /^    at Object\.eval \(eval at <anonymous> \(.*\/line1\.js:1001:101\)/,
    /^    at Object\.exports\.test \(.*\/line1\.js:1001:101\)$/
  ]);
});

it('eval inside function', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'function foo() {',
    '  eval("throw new Error(\'test\')");',
    '}',
    'foo();'
  ], [
    'Error: test',
    /^    at eval \(eval at foo \(.*\/line2\.js:1002:102\)/,
    /^    at foo \(.*\/line2\.js:1002:102\)/,
    /^    at Object\.exports\.test \(.*\/line4\.js:1004:104\)$/
  ]);
});

it('eval with sourceURL', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'eval("throw new Error(\'test\')//@ sourceURL=sourceURL.js");'
  ], [
    'Error: test',
    '    at Object.eval (sourceURL.js:1:7)',
    /^    at Object\.exports\.test \(.*\/line1\.js:1001:101\)$/
  ]);
});

it('eval with sourceURL inside eval', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'eval("eval(\'throw new Error(\\"test\\")//@ sourceURL=sourceURL.js\')");'
  ], [
    'Error: test',
    '    at Object.eval (sourceURL.js:1:7)',
    /^    at Object\.eval \(eval at <anonymous> \(.*\/line1\.js:1001:101\)/,
    /^    at Object\.exports\.test \(.*\/line1\.js:1001:101\)$/
  ]);
});

it('function constructor', function() {
  compareStackTrace(createMultiLineSourceMap(), [
    'throw new Function(")");'
  ], [
    'SyntaxError: Unexpected token )',
    /^    at Object\.Function \((?:unknown source|<anonymous>)\)$/,
    /^    at Object\.exports\.test \(.*\/line1\.js:1001:101\)$/,
  ]);
});

it('throw with empty source map', function() {
  compareStackTrace(createEmptySourceMap(), [
    'throw new Error("test");'
  ], [
    'Error: test',
    /^    at Object\.exports\.test \(.*\/.generated.js:1:96\)$/
  ]);
});

it('throw with source map with gap', function() {
  compareStackTrace(createSourceMapWithGap(), [
    'throw new Error("test");'
  ], [
    'Error: test',
    /^    at Object\.exports\.test \(.*\/.generated.js:1:96\)$/
  ]);
});

it('sourcesContent with data URL', function() {
  compareStackTrace(createMultiLineSourceMapWithSourcesContent(), [
    'throw new Error("test");'
  ], [
    'Error: test',
    /^    at Object\.exports\.test \(.*\/original.js:1001:5\)$/
  ]);
});

it('default options', function(done) {
  compareStdout(done, createSingleLineSourceMap(), [
    '',
    'function foo() { throw new Error("this is the error"); }',
    'require("./source-map-support").install();',
    'process.nextTick(foo);',
    'process.nextTick(function() { process.exit(1); });'
  ], [
    /\/.original\.js:1$/,
    'this is the original code',
    '^',
    'Error: this is the error',
    /^    at foo \(.*\/.original\.js:1:1\)$/
  ]);
});

it('handleUncaughtExceptions is true', function(done) {
  compareStdout(done, createSingleLineSourceMap(), [
    '',
    'function foo() { throw new Error("this is the error"); }',
    'require("./source-map-support").install({ handleUncaughtExceptions: true });',
    'process.nextTick(foo);'
  ], [
    /\/.original\.js:1$/,
    'this is the original code',
    '^',
    'Error: this is the error',
    /^    at foo \(.*\/.original\.js:1:1\)$/
  ]);
});

it('handleUncaughtExceptions is false', function(done) {
  compareStdout(done, createSingleLineSourceMap(), [
    '',
    'function foo() { throw new Error("this is the error"); }',
    'require("./source-map-support").install({ handleUncaughtExceptions: false });',
    'process.nextTick(foo);'
  ], [
    /\/.generated.js:2$/,
    'function foo() { throw new Error("this is the error"); }',
    '                       ^',
    'Error: this is the error',
    /^    at foo \(.*\/.original\.js:1:1\)$/
  ]);
});

it('default options with empty source map', function(done) {
  compareStdout(done, createEmptySourceMap(), [
    '',
    'function foo() { throw new Error("this is the error"); }',
    'require("./source-map-support").install();',
    'process.nextTick(foo);'
  ], [
    /\/.generated.js:2$/,
    'function foo() { throw new Error("this is the error"); }',
    '                       ^',
    'Error: this is the error',
    /^    at foo \(.*\/.generated.js:2:24\)$/
  ]);
});

it('default options with source map with gap', function(done) {
  compareStdout(done, createSourceMapWithGap(), [
    '',
    'function foo() { throw new Error("this is the error"); }',
    'require("./source-map-support").install();',
    'process.nextTick(foo);'
  ], [
    /\/.generated.js:2$/,
    'function foo() { throw new Error("this is the error"); }',
    '                       ^',
    'Error: this is the error',
    /^    at foo \(.*\/.generated.js:2:24\)$/
  ]);
});

it('specifically requested error source', function(done) {
  compareStdout(done, createSingleLineSourceMap(), [
    '',
    'function foo() { throw new Error("this is the error"); }',
    'var sms = require("./source-map-support");',
    'sms.install({ handleUncaughtExceptions: false });',
    'process.on("uncaughtException", function (e) { console.log("SRC:" + sms.getErrorSource(e)); });',
    'process.nextTick(foo);'
  ], [
    'SRC:',
    /\/.original.js:1$/,
    'this is the original code',
    '^'
  ]);
});

it('sourcesContent', function(done) {
  compareStdout(done, createMultiLineSourceMapWithSourcesContent(), [
    '',
    'function foo() { throw new Error("this is the error"); }',
    'require("./source-map-support").install();',
    'process.nextTick(foo);',
    'process.nextTick(function() { process.exit(1); });'
  ], [
    /\/original\.js:1002$/,
    '    line 2',
    '    ^',
    'Error: this is the error',
    /^    at foo \(.*\/original\.js:1002:5\)$/
  ]);
});

it('missing source maps should also be cached', function(done) {
  compareStdout(done, createSingleLineSourceMap(), [
    '',
    'var count = 0;',
    'function foo() {',
    '  console.log(new Error("this is the error").stack.split("\\n").slice(0, 2).join("\\n"));',
    '}',
    'require("./source-map-support").install({',
    '  retrieveSourceMap: function(name) {',
    '    if (/\\.generated.js$/.test(name)) count++;',
    '    return null;',
    '  }',
    '});',
    'process.nextTick(foo);',
    'process.nextTick(foo);',
    'process.nextTick(function() { console.log(count); });',
  ], [
    'Error: this is the error',
    /^    at foo \(.*\/.generated.js:4:15\)$/,
    'Error: this is the error',
    /^    at foo \(.*\/.generated.js:4:15\)$/,
    '1', // The retrieval should only be attempted once
  ]);
});
