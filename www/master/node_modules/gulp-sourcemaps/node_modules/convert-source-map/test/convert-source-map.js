'use strict';
/*jshint asi: true */

var test = require('tap').test
  , generator = require('inline-source-map')
  , convert = require('..')

var gen = generator()
    .addMappings('foo.js', [{ original: { line: 2, column: 3 } , generated: { line: 5, column: 10 } }], { line: 5 })
    .addGeneratedMappings('bar.js', 'var a = 2;\nconsole.log(a)', { line: 23, column: 22 })

  , base64 = gen.base64Encode()
  , comment = gen.inlineMappingUrl()
  , json = gen.toString()
  , obj = JSON.parse(json)

test('different formats', function (t) {

  t.equal(convert.fromComment(comment).toComment(), comment, 'comment -> comment')
  t.equal(convert.fromComment(comment).toBase64(), base64, 'comment -> base64')
  t.equal(convert.fromComment(comment).toJSON(), json, 'comment -> json')
  t.deepEqual(convert.fromComment(comment).toObject(), obj, 'comment -> object')

  t.equal(convert.fromBase64(base64).toBase64(), base64, 'base64 -> base64')
  t.equal(convert.fromBase64(base64).toComment(), comment, 'base64 -> comment')
  t.equal(convert.fromBase64(base64).toJSON(), json, 'base64 -> json')
  t.deepEqual(convert.fromBase64(base64).toObject(), obj, 'base64 -> object')

  t.equal(convert.fromJSON(json).toJSON(), json, 'json -> json')
  t.equal(convert.fromJSON(json).toBase64(), base64, 'json -> base64')
  t.equal(convert.fromJSON(json).toComment(), comment, 'json -> comment')
  t.deepEqual(convert.fromJSON(json).toObject(), obj, 'json -> object')
  t.end()
})

test('to object returns a copy', function (t) {
  var c = convert.fromJSON(json)
  var o = c.toObject()
  o.version = '99';
  t.equal(c.toObject().version, 3, 'setting property on returned object does not affect original')
  t.end()
})

test('to multi-line map', function (t) {
  var c = convert.fromObject(obj);
  var s = c.toComment({ multiline: true });
  t.similar(s, /^\/\*# sourceMappingURL=.+ \*\/$/);
  t.end();
})

test('from source', function (t) {
  var foo = [
      'function foo() {'
    , ' console.log("hello I am foo");'
    , ' console.log("who are you");'
    , '}'
    , ''
    , 'foo();'
    , ''
    ].join('\n')
  , map = '//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlcyI6WyJmdW5jdGlvbiBmb28oKSB7XG4gY29uc29sZS5sb2coXCJoZWxsbyBJIGFtIGZvb1wiKTtcbiBjb25zb2xlLmxvZyhcIndobyBhcmUgeW91XCIpO1xufVxuXG5mb28oKTtcbiJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSJ9'
  , otherMap = '//# sourceMappingURL=data:application/json;base64,otherZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlcyI6WyJmdW5jdGlvbiBmb28oKSB7XG4gY29uc29sZS5sb2coXCJoZWxsbyBJIGFtIGZvb1wiKTtcbiBjb25zb2xlLmxvZyhcIndobyBhcmUgeW91XCIpO1xufVxuXG5mb28oKTtcbiJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSJ9'

  function getComment(src) {
    var map = convert.fromSource(src);
    return map ? map.toComment() : null;
  }

  t.equal(getComment(foo), null, 'no comment returns null')
  t.equal(getComment(foo + map), map, 'beginning of last line')
  t.equal(getComment(foo + '    ' +  map), map, 'indented of last line')
  t.equal(getComment(foo + '   ' + map + '\n\n'), map, 'indented on last non empty line')
  t.equal(getComment(foo + map + '\nconsole.log("more code");\nfoo()\n'), map, 'in the middle of code')
  t.equal(getComment(foo + otherMap + '\n' +  map), map, 'finds last map in source')
  t.end()
})

test('from source with a large source', function (t) {
  var foo = [
      'function foo() {'
    , ' console.log("hello I am foo");'
    , ' console.log("who are you");'
    , '}'
    , ''
    , 'foo();'
    , ''
    ].join('\n')
  , map = '//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlcyI6WyJmdW5jdGlvbiBmb28oKSB7XG4gY29uc29sZS5sb2coXCJoZWxsbyBJIGFtIGZvb1wiKTtcbiBjb25zb2xlLmxvZyhcIndobyBhcmUgeW91XCIpO1xufVxuXG5mb28oKTtcbiJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSJ9'
  , otherMap = '//# sourceMappingURL=data:application/json;base64,otherZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlcyI6WyJmdW5jdGlvbiBmb28oKSB7XG4gY29uc29sZS5sb2coXCJoZWxsbyBJIGFtIGZvb1wiKTtcbiBjb25zb2xlLmxvZyhcIndobyBhcmUgeW91XCIpO1xufVxuXG5mb28oKTtcbiJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSJ9'

  function getComment(src) {
    var map = convert.fromSource(src, true);
    return map ? map.toComment() : null;
  }

  t.equal(getComment(foo), null, 'no comment returns null')
  t.equal(getComment(foo + map), map, 'beginning of last line')
  t.equal(getComment(foo + '    ' +  map), map, 'indented of last line')
  t.equal(getComment(foo + '   ' + map + '\n\n'), map, 'indented on last non empty line')
  t.equal(getComment(foo + map + '\nconsole.log("more code");\nfoo()\n'), map, 'in the middle of code')
  t.equal(getComment(foo + otherMap + '\n' +  map), map, 'finds last map in source')
  t.end()
})

test('remove comments', function (t) {
  var foo = [
      'function foo() {'
    , ' console.log("hello I am foo");'
    , ' console.log("who are you");'
    , '}'
    , ''
    , 'foo();'
    , ''
    ].join('\n')
    // this one is old spec on purpose
  , map = '//@ sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlcyI6WyJmdW5jdGlvbiBmb28oKSB7XG4gY29uc29sZS5sb2coXCJoZWxsbyBJIGFtIGZvb1wiKTtcbiBjb25zb2xlLmxvZyhcIndobyBhcmUgeW91XCIpO1xufVxuXG5mb28oKTtcbiJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSJ9'
  , otherMap = '//# sourceMappingURL=data:application/json;base64,ZXJzaW9uIjozLCJmaWxlIjoiIiwic291cmNlcyI6WyJmdW5jdGlvbiBmb28oKSB7XG4gY29uc29sZS5sb2coXCJoZWxsbyBJIGFtIGZvb1wiKTtcbiBjb25zb2xlLmxvZyhcIndobyBhcmUgeW91XCIpO1xufVxuXG5mb28oKTtcbiJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSJ9'
  , extraCode = '\nconsole.log("more code");\nfoo()\n'

  t.equal(convert.removeComments(foo + map), foo, 'from last line')
  t.equal(convert.removeComments(foo + map + extraCode), foo + extraCode, 'from the middle of code')
  t.equal(convert.removeComments(foo + otherMap + extraCode + map), foo + extraCode, 'multiple comments from the middle of code')
  t.end()
})

test('remove map file comments', function (t) {
  var foo = [
      'function foo() {'
    , ' console.log("hello I am foo");'
    , ' console.log("who are you");'
    , '}'
    , ''
    , 'foo();'
    , ''
    ].join('\n')
  , fileMap1 = '//# sourceMappingURL=foo.js.map'
  , fileMap2 = '/*# sourceMappingURL=foo.js.map */';

  t.equal(convert.removeMapFileComments(foo + fileMap1), foo, '// style filemap comment')
  t.equal(convert.removeMapFileComments(foo + fileMap2), foo, '/* */ style filemap comment')
  t.end()
})

test('pretty json', function (t) {
  var mod = convert.fromJSON(json).toJSON(2)
    , expected = JSON.stringify(obj, null, 2);

  t.equal(
      mod
    , expected
    , 'pretty prints json when space is given')
  t.end()
})

test('adding properties', function (t) {
  var mod = convert
    .fromJSON(json)
    .addProperty('foo', 'bar')
    .toJSON()
    , expected = JSON.parse(json);
    expected.foo = 'bar';
  t.equal(
      mod
    , JSON.stringify(expected)
    , 'includes added property'
  )
  t.end()
})

test('setting properties', function (t) {
  var mod = convert
    .fromJSON(json)
    .setProperty('version', '2')
    .setProperty('mappings', ';;;UACG')
    .setProperty('should add', 'this')
    .toJSON()
    , expected = JSON.parse(json);
    expected.version = '2';
    expected.mappings = ';;;UACG';
    expected['should add'] = 'this';
  t.equal(
      mod
    , JSON.stringify(expected)
    , 'includes new property and changes existing properties'
  )
  t.end()
})

test('getting properties', function (t) {
  var sm = convert.fromJSON(json)

  t.equal(sm.getProperty('version'), 3, 'gets version')
  t.deepEqual(sm.getProperty('sources'), ['foo.js', 'bar.js'], 'gets sources')
  t.end()
})
