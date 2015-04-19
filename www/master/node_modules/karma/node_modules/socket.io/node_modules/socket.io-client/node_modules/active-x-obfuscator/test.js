var activeXObfuscator = require('./index');
var assert            = require('assert');

var OBFUSCATED_ACTIVE_X_OBJECT = activeXObfuscator.OBFUSCATED_ACTIVE_X_OBJECT;
var OBFUSCATED_ACTIVE_X        = activeXObfuscator.OBFUSCATED_ACTIVE_X;

var input =
  "foo(new ActiveXObject('Microsoft.XMLHTTP'))";
var expected =
  "foo(new window[" + OBFUSCATED_ACTIVE_X_OBJECT + "]('Microsoft.XMLHTTP'))";
assert.equal(activeXObfuscator(input), expected);

var input =
  "var foo = 'ActiveXObject';";
var expected =
  "var foo = " + OBFUSCATED_ACTIVE_X_OBJECT + ";";
assert.equal(activeXObfuscator(input), expected);

var input =
  'var foo = "ActiveXObject";';
var expected =
  "var foo = " + OBFUSCATED_ACTIVE_X_OBJECT + ";";
assert.equal(activeXObfuscator(input), expected);

var input =
  'var foo = o.ActiveXObject;';
var expected =
  "var foo = o[" + OBFUSCATED_ACTIVE_X_OBJECT + "];";
assert.equal(activeXObfuscator(input), expected);

var input =
  'var foo = "ActiveX";';
var expected =
  "var foo = " + OBFUSCATED_ACTIVE_X + ";";
assert.equal(activeXObfuscator(input), expected);

var input =
  "var foo = 'ActiveX';";
var expected =
  "var foo = " + OBFUSCATED_ACTIVE_X + ";";
assert.equal(activeXObfuscator(input), expected);

var input =
  "var foo; // ActiveX is cool";
var expected =
  "var foo; // Ac...eX is cool";
assert.equal(activeXObfuscator(input), expected);

var input =
  "var foo = 'ActiveX is cool';";
assert.throws(function() {
  activeXObfuscator(input);
}, /Unknown ActiveX occurence/);
