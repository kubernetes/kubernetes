'use strict';

var path = require('path');

var globule = require('../lib/globule.js');

/*
  ======== A Handy Little Nodeunit Reference ========
  https://github.com/caolan/nodeunit

  Test methods:
    test.expect(numAssertions)
    test.done()
  Test assertions:
    test.ok(value, [message])
    test.equal(actual, expected, [message])
    test.notEqual(actual, expected, [message])
    test.deepEqual(actual, expected, [message])
    test.notDeepEqual(actual, expected, [message])
    test.strictEqual(actual, expected, [message])
    test.notStrictEqual(actual, expected, [message])
    test.throws(block, [error], [message])
    test.doesNotThrow(block, [error], [message])
    test.ifError(value)
*/

exports['match'] = {
  'empty set': function(test) {
    test.expect(6);
    // Should return empty set if a required argument is missing or an empty set.
    test.deepEqual(globule.match(null, 'foo.js'), [], 'should return empty set.');
    test.deepEqual(globule.match('*.js', null), [], 'should return empty set.');
    test.deepEqual(globule.match([], 'foo.js'), [], 'should return empty set.');
    test.deepEqual(globule.match('*.js', []), [], 'should return empty set.');
    test.deepEqual(globule.match(null, ['foo.js']), [], 'should return empty set.');
    test.deepEqual(globule.match(['*.js'], null), [], 'should return empty set.');
    test.done();
  },
  'basic matching': function(test) {
    test.expect(6);
    test.deepEqual(globule.match('*.js', 'foo.js'), ['foo.js'], 'should match correctly.');
    test.deepEqual(globule.match('*.js', ['foo.js']), ['foo.js'], 'should match correctly.');
    test.deepEqual(globule.match('*.js', ['foo.js', 'bar.css']), ['foo.js'], 'should match correctly.');
    test.deepEqual(globule.match(['*.js', '*.css'], 'foo.js'), ['foo.js'], 'should match correctly.');
    test.deepEqual(globule.match(['*.js', '*.css'], ['foo.js']), ['foo.js'], 'should match correctly.');
    test.deepEqual(globule.match(['*.js', '*.css'], ['foo.js', 'bar.css']), ['foo.js', 'bar.css'], 'should match correctly.');
    test.done();
  },
  'no matches': function(test) {
    test.expect(2);
    test.deepEqual(globule.match('*.js', 'foo.css'), [], 'should fail to match.');
    test.deepEqual(globule.match('*.js', ['foo.css', 'bar.css']), [], 'should fail to match.');
    test.done();
  },
  'unique': function(test) {
    test.expect(2);
    test.deepEqual(globule.match('*.js', ['foo.js', 'foo.js']), ['foo.js'], 'should return a uniqued set.');
    test.deepEqual(globule.match(['*.js', '*.*'], ['foo.js', 'foo.js']), ['foo.js'], 'should return a uniqued set.');
    test.done();
  },
  'flatten': function(test) {
    test.expect(1);
    test.deepEqual(globule.match([['*.js', '*.css'], ['*.*', '*.js']], ['foo.js', 'bar.css']),
      ['foo.js', 'bar.css'],
      'should process nested pattern arrays correctly.');
    test.done();
  },
  'exclusion': function(test) {
    test.expect(5);
    test.deepEqual(globule.match(['!*.js'], ['foo.js', 'bar.js']), [], 'solitary exclusion should match nothing');
    test.deepEqual(globule.match(['*.js', '!*.js'], ['foo.js', 'bar.js']), [], 'exclusion should cancel match');
    test.deepEqual(globule.match(['*.js', '!f*.js'], ['foo.js', 'bar.js', 'baz.js']),
      ['bar.js', 'baz.js'],
      'partial exclusion should partially cancel match');
    test.deepEqual(globule.match(['*.js', '!*.js', 'b*.js'], ['foo.js', 'bar.js', 'baz.js']),
      ['bar.js', 'baz.js'],
      'inclusion / exclusion order matters');
    test.deepEqual(globule.match(['*.js', '!f*.js', '*.js'], ['foo.js', 'bar.js', 'baz.js']),
      ['bar.js', 'baz.js', 'foo.js'],
      'inclusion / exclusion order matters');
    test.done();
  },
  'options.matchBase': function(test) {
    test.expect(2);
    test.deepEqual(globule.match('*.js', ['foo.js', 'bar', 'baz/xyz.js'], {matchBase: true}),
      ['foo.js', 'baz/xyz.js'],
      'should matchBase (minimatch) when specified.');
    test.deepEqual(globule.match('*.js', ['foo.js', 'bar', 'baz/xyz.js']),
      ['foo.js'],
      'should not matchBase (minimatch) by default.');
    test.done();
  },
};

exports['isMatch'] = {
  'basic matching': function(test) {
    test.expect(6);
    test.ok(globule.isMatch('*.js', 'foo.js'), 'should match correctly.');
    test.ok(globule.isMatch('*.js', ['foo.js']), 'should match correctly.');
    test.ok(globule.isMatch('*.js', ['foo.js', 'bar.css']), 'should match correctly.');
    test.ok(globule.isMatch(['*.js', '*.css'], 'foo.js'), 'should match correctly.');
    test.ok(globule.isMatch(['*.js', '*.css'], ['foo.js']), 'should match correctly.');
    test.ok(globule.isMatch(['*.js', '*.css'], ['foo.js', 'bar.css']), 'should match correctly.');
    test.done();
  },
  'no matches': function(test) {
    test.expect(6);
    test.ok(!globule.isMatch('*.js', 'foo.css'), 'should fail to match.');
    test.ok(!globule.isMatch('*.js', ['foo.css', 'bar.css']), 'should fail to match.');
    test.ok(!globule.isMatch(null, 'foo.css'), 'should fail to match.');
    test.ok(!globule.isMatch('*.js', null), 'should fail to match.');
    test.ok(!globule.isMatch([], 'foo.css'), 'should fail to match.');
    test.ok(!globule.isMatch('*.js', []), 'should fail to match.');
    test.done();
  },
  'options.matchBase': function(test) {
    test.expect(2);
    test.ok(globule.isMatch('*.js', ['baz/xyz.js'], {matchBase: true}), 'should matchBase (minimatch) when specified.');
    test.ok(!globule.isMatch('*.js', ['baz/xyz.js']), 'should not matchBase (minimatch) by default.');
    test.done();
  },
};

exports['find'] = {
  setUp: function(done) {
    this.cwd = process.cwd();
    process.chdir('test/fixtures/expand');
    done();
  },
  tearDown: function(done) {
    process.chdir(this.cwd);
    done();
  },
  'basic matching': function(test) {
    test.expect(5);
    test.deepEqual(globule.find('**/*.js'), ['js/bar.js', 'js/foo.js'], 'single pattern argument should match.');
    test.deepEqual(globule.find('**/*.js', '**/*.css'),
      ['js/bar.js', 'js/foo.js', 'css/baz.css', 'css/qux.css'],
      'multiple pattern arguments should match.');
    test.deepEqual(globule.find(['**/*.js', '**/*.css']),
      ['js/bar.js', 'js/foo.js', 'css/baz.css', 'css/qux.css'],
      'array of patterns should match.');
    test.deepEqual(globule.find([['**/*.js'], [['**/*.css', 'js/*.js']]]),
      ['js/bar.js', 'js/foo.js', 'css/baz.css', 'css/qux.css'],
      'array of arrays of patterns should be flattened.');
    test.deepEqual(globule.find('*.xyz'), [], 'bad pattern should fail to match.');
    test.done();
  },
  'unique': function(test) {
    test.expect(4);
    test.deepEqual(globule.find('**/*.js', 'js/*.js'),
      ['js/bar.js', 'js/foo.js'],
      'file list should be uniqed.');
    test.deepEqual(globule.find('**/*.js', '**/*.css', 'js/*.js'), ['js/bar.js', 'js/foo.js',
      'css/baz.css', 'css/qux.css'],
      'file list should be uniqed.');
    test.deepEqual(globule.find('js', 'js/'),
      ['js', 'js/'],
      'mixed non-ending-/ and ending-/ dirs will not be uniqed by default.');
    test.deepEqual(globule.find('js', 'js/', {mark: true}),
      ['js/'],
      'mixed non-ending-/ and ending-/ dirs will be uniqed when "mark" is specified.');
    test.done();
  },
  'file order': function(test) {
    test.expect(5);
    var actual = globule.find('**/*.{js,css}');
    var expected = ['css/baz.css', 'css/qux.css', 'js/bar.js', 'js/foo.js'];
    test.deepEqual(actual, expected, 'should select 4 files in this order, by default.');

    actual = globule.find('js/foo.js', 'js/bar.js', '**/*.{js,css}');
    expected = ['js/foo.js', 'js/bar.js', 'css/baz.css', 'css/qux.css'];
    test.deepEqual(actual, expected, 'specifically-specified-up-front file order should be maintained.');

    actual = globule.find('js/bar.js', 'js/foo.js', '**/*.{js,css}');
    expected = ['js/bar.js', 'js/foo.js', 'css/baz.css', 'css/qux.css'];
    test.deepEqual(actual, expected, 'specifically-specified-up-front file order should be maintained.');

    actual = globule.find('**/*.{js,css}', '!css/qux.css', 'css/qux.css');
    expected = ['css/baz.css', 'js/bar.js', 'js/foo.js', 'css/qux.css'];
    test.deepEqual(actual, expected, 'if a file is excluded and then re-added, it should be added at the end.');

    actual = globule.find('js/foo.js', '**/*.{js,css}', '!css/qux.css', 'css/qux.css');
    expected = ['js/foo.js', 'css/baz.css', 'js/bar.js', 'css/qux.css'];
    test.deepEqual(actual, expected, 'should be able to combine specified-up-front and excluded/added-at-end.');
    test.done();
  },
  'exclusion': function(test) {
    test.expect(8);
    test.deepEqual(globule.find(['!js/*.js']), [], 'solitary exclusion should match nothing');
    test.deepEqual(globule.find(['js/bar.js','!js/bar.js']), [], 'exclusion should negate match');
    test.deepEqual(globule.find(['**/*.js', '!js/foo.js']),
      ['js/bar.js'],
      'should omit single file from matched set');
    test.deepEqual(globule.find(['!js/foo.js', '**/*.js']),
      ['js/bar.js', 'js/foo.js'],
      'inclusion / exclusion order matters');
    test.deepEqual(globule.find(['**/*.js', '**/*.css', '!js/bar.js', '!css/baz.css']),
      ['js/foo.js','css/qux.css'],
      'multiple exclusions should be removed from the set');
    test.deepEqual(globule.find(['**/*.js', '**/*.css', '!**/*.css']),
      ['js/bar.js', 'js/foo.js'],
      'excluded wildcards should be removed from the matched set');
    test.deepEqual(globule.find(['js/bar.js', 'js/foo.js', 'css/baz.css', 'css/qux.css', '!**/b*.*']),
      ['js/foo.js', 'css/qux.css'],
      'different pattern for exclusion should still work');
    test.deepEqual(globule.find(['js/bar.js', '!**/b*.*', 'js/foo.js', 'css/baz.css', 'css/qux.css']),
      ['js/foo.js', 'css/baz.css', 'css/qux.css'],
      'inclusion / exclusion order matters');
    test.done();
  },
  'options.mark': function(test) {
    test.expect(4);
    test.deepEqual(globule.find('**d*/**'), [
      'deep',
      'deep/deep.txt',
      'deep/deeper',
      'deep/deeper/deeper.txt',
      'deep/deeper/deepest',
      'deep/deeper/deepest/deepest.txt'], 'should match files and directories.');
    test.deepEqual(globule.find('**d*/**/'), [
      'deep/',
      'deep/deeper/',
      'deep/deeper/deepest/'], 'trailing / in pattern should match directories only, matches end in /.');
    test.deepEqual(globule.find('**d*/**', {mark: true}), [
      'deep/',
      'deep/deep.txt',
      'deep/deeper/',
      'deep/deeper/deeper.txt',
      'deep/deeper/deepest/',
      'deep/deeper/deepest/deepest.txt'], 'the minimatch "mark" option ensures directories end in /.');
    test.deepEqual(globule.find('**d*/**/', {mark: true}), [
      'deep/',
      'deep/deeper/',
      'deep/deeper/deepest/'], 'the minimatch "mark" option should not remove trailing / from matched paths.');
    test.done();
  },
  'options.filter': function(test) {
    test.expect(5);
    test.deepEqual(globule.find('**d*/**', {filter: 'isFile'}), [
      'deep/deep.txt',
      'deep/deeper/deeper.txt',
      'deep/deeper/deepest/deepest.txt'
    ], 'should match files only.');
    test.deepEqual(globule.find('**d*/**', {filter: 'isDirectory'}), [
      'deep',
      'deep/deeper',
      'deep/deeper/deepest'
    ], 'should match directories only.');
    test.deepEqual(globule.find('**', {
      arbitraryProp: /deepest/,
      filter: function(filepath, options) {
        return options.arbitraryProp.test(filepath);
      }
    }), [
      'deep/deeper/deepest',
      'deep/deeper/deepest/deepest.txt',
    ], 'should filter arbitrarily.');
    test.deepEqual(globule.find('js', 'css', {filter: 'isFile'}), [], 'should fail to match.');
    test.deepEqual(globule.find('**/*.js', {filter: 'isDirectory'}), [], 'should fail to match.');
    test.done();
  },
  'options.matchBase': function(test) {
    test.expect(3);
    test.deepEqual(globule.find('*.js'), [], 'should not matchBase (minimatch) by default.');
    test.deepEqual(globule.find('*.js', {matchBase: true}),
      ['js/bar.js', 'js/foo.js'],
      'matchBase option should be passed through to minimatch.');
    test.deepEqual(globule.find('*.js', '*.css', {matchBase: true}),
      ['js/bar.js', 'js/foo.js', 'css/baz.css', 'css/qux.css'],
      'matchBase option should be passed through to minimatch.');
    test.done();
  },
  'options.srcBase': function(test) {
    test.expect(5);
    test.deepEqual(globule.find(['**/deep*.txt'], {srcBase: 'deep'}),
      ['deep.txt', 'deeper/deeper.txt', 'deeper/deepest/deepest.txt'],
      'should find paths matching pattern relative to srcBase.');
    test.deepEqual(globule.find(['**/deep*.txt'], {cwd: 'deep'}),
      ['deep.txt', 'deeper/deeper.txt', 'deeper/deepest/deepest.txt'],
      'cwd and srcBase should do the same thing.');
    test.deepEqual(globule.find(['**/deep*'], {srcBase: 'deep', filter: 'isFile'}),
      ['deep.txt', 'deeper/deeper.txt', 'deeper/deepest/deepest.txt'],
      'srcBase should not prevent filtering.');
    test.deepEqual(globule.find(['**/deep*'], {srcBase: 'deep', filter: 'isDirectory'}),
      ['deeper', 'deeper/deepest'],
      'srcBase should not prevent filtering.');
    test.deepEqual(globule.find(['**/deep*.txt', '!**/deeper**'], {srcBase: 'deep'}),
      ['deep.txt', 'deeper/deepest/deepest.txt'],
      'srcBase should not prevent exclusions.');
    test.done();
  },
  'options.prefixBase': function(test) {
    test.expect(2);
    test.deepEqual(globule.find(['**/deep*.txt'], {srcBase: 'deep', prefixBase: false}),
      ['deep.txt', 'deeper/deeper.txt', 'deeper/deepest/deepest.txt'],
      'should not prefix srcBase to returned paths.');
    test.deepEqual(globule.find(['**/deep*.txt'], {srcBase: 'deep', prefixBase: true}),
      ['deep/deep.txt', 'deep/deeper/deeper.txt', 'deep/deeper/deepest/deepest.txt'],
      'should prefix srcBase to returned paths.');
    test.done();
  },
  'options.nonull': function(test) {
    test.expect(3);
    test.deepEqual(globule.find(['*omg*'], {nonull: true}),
      ['*omg*'],
      'non-matching patterns should be returned in result set.');
    test.deepEqual(globule.find(['js/a*', 'js/b*', 'js/c*'], {nonull: true}),
      ['js/a*', 'js/bar.js', 'js/c*'],
      'non-matching patterns should be returned in result set.');
    test.deepEqual(globule.find(['js/foo.js', 'js/bar.js', 'js/nonexistent.js'], {nonull: true}),
      ['js/foo.js', 'js/bar.js', 'js/nonexistent.js'],
      'non-matching filenames should be returned in result set.');
    test.done();
  },
};

exports['mapping'] = {
  'basic mapping': function(test) {
    test.expect(1);

    var actual = globule.mapping(['a.txt', 'b.txt', 'c.txt']);
    var expected = [
      {dest: 'a.txt', src: ['a.txt']},
      {dest: 'b.txt', src: ['b.txt']},
      {dest: 'c.txt', src: ['c.txt']},
    ];
    test.deepEqual(actual, expected, 'default options should create same-to-same src-dest mappings.');

    test.done();
  },
  'options.srcBase': function(test) {
    test.expect(2);
    var actual, expected;
    actual = globule.mapping(['a.txt', 'bar/b.txt', 'bar/baz/c.txt'], {srcBase: 'foo'});
    expected = [
      {dest: 'a.txt', src: ['foo/a.txt']},
      {dest: 'bar/b.txt', src: ['foo/bar/b.txt']},
      {dest: 'bar/baz/c.txt', src: ['foo/bar/baz/c.txt']},
    ];
    test.deepEqual(actual, expected, 'srcBase should be prefixed to src paths (no trailing /).');

    actual = globule.mapping(['a.txt', 'bar/b.txt', 'bar/baz/c.txt'], {srcBase: 'foo/'});
    test.deepEqual(actual, expected, 'srcBase should be prefixed to src paths (trailing /).');

    test.done();
  },
  'options.destBase': function(test) {
    test.expect(2);
    var actual, expected;

    actual = globule.mapping(['a.txt', 'bar/b.txt', 'bar/baz/c.txt'], {destBase: 'dest'});
    expected = [
      {dest: 'dest/a.txt', src: ['a.txt']},
      {dest: 'dest/bar/b.txt', src: ['bar/b.txt']},
      {dest: 'dest/bar/baz/c.txt', src: ['bar/baz/c.txt']},
    ];
    test.deepEqual(actual, expected, 'destBase should be prefixed to dest paths (no trailing /).');

    actual = globule.mapping(['a.txt', 'bar/b.txt', 'bar/baz/c.txt'], {destBase: 'dest/'});
    test.deepEqual(actual, expected, 'destBase should be prefixed to dest paths (trailing /).');

    test.done();
  },
  'options.flatten': function(test) {
    test.expect(1);
    var actual, expected;

    actual = globule.mapping(['a.txt', 'bar/b.txt', 'bar/baz/c.txt'], {flatten: true});
    expected = [
      {dest: 'a.txt', src: ['a.txt']},
      {dest: 'b.txt', src: ['bar/b.txt']},
      {dest: 'c.txt', src: ['bar/baz/c.txt']},
    ];
    test.deepEqual(actual, expected, 'flatten should remove all src path parts from dest.');

    test.done();
  },
  'options.flatten + options.destBase': function(test) {
    test.expect(1);
    var actual, expected;

    actual = globule.mapping(['a.txt', 'bar/b.txt', 'bar/baz/c.txt'], {destBase: 'dest', flatten: true});
    expected = [
      {dest: 'dest/a.txt', src: ['a.txt']},
      {dest: 'dest/b.txt', src: ['bar/b.txt']},
      {dest: 'dest/c.txt', src: ['bar/baz/c.txt']},
    ];
    test.deepEqual(actual, expected, 'flatten and destBase should work together.');

    test.done();
  },
  'options.ext': function(test) {
    test.expect(1);
    var actual, expected;

    actual = globule.mapping(['x/a.js', 'x.y/b.min.js', 'x.y/z.z/c'], {ext: '.foo'});
    expected = [
      {dest: 'x/a.foo', src: ['x/a.js']},
      {dest: 'x.y/b.foo', src: ['x.y/b.min.js']},
      {dest: 'x.y/z.z/c.foo', src: ['x.y/z.z/c']},
    ];
    test.deepEqual(actual, expected, 'by default, ext should replace everything after the first dot in the filename.');

    test.done();
  },
  'options.extDot': function(test) {
    test.expect(2);
    var actual, expected;

    actual = globule.mapping(['x/a.js', 'x.y/b.bbb.min.js', 'x.y/z.z/c'], {ext: '.foo', extDot: 'first'});
    expected = [
      {dest: 'x/a.foo', src: ['x/a.js']},
      {dest: 'x.y/b.foo', src: ['x.y/b.bbb.min.js']},
      {dest: 'x.y/z.z/c.foo', src: ['x.y/z.z/c']},
    ];
    test.deepEqual(actual, expected, 'extDot of "first" should replace everything after the first dot in the filename.');

    actual = globule.mapping(['x/a.js', 'x.y/b.bbb.min.js', 'x.y/z.z/c'], {ext: '.foo', extDot: 'last'});
    expected = [
      {dest: 'x/a.foo', src: ['x/a.js']},
      {dest: 'x.y/b.bbb.min.foo', src: ['x.y/b.bbb.min.js']},
      {dest: 'x.y/z.z/c.foo', src: ['x.y/z.z/c']},
    ];
    test.deepEqual(actual, expected, 'extDot of "last" should replace everything after the last dot in the filename.');

    test.done();
  },
  'options.rename': function(test) {
    test.expect(1);
    var actual, expected;
    actual = globule.mapping(['a.txt', 'bar/b.txt', 'bar/baz/c.txt'], {
      arbitraryProp: 'FOO',
      rename: function(dest, options) {
        return path.join(options.arbitraryProp, dest.toUpperCase());
      }
    });
    expected = [
      {dest: 'FOO/A.TXT', src: ['a.txt']},
      {dest: 'FOO/BAR/B.TXT', src: ['bar/b.txt']},
      {dest: 'FOO/BAR/BAZ/C.TXT', src: ['bar/baz/c.txt']},
    ];
    test.deepEqual(actual, expected, 'allow arbitrary renaming of files.');

    test.done();
  },
};

exports['findMapping'] = {
  setUp: function(done) {
    this.cwd = process.cwd();
    process.chdir('test/fixtures');
    done();
  },
  tearDown: function(done) {
    process.chdir(this.cwd);
    done();
  },
  'basic matching': function(test) {
    test.expect(2);

    var actual = globule.findMapping(['expand/**/*.txt']);
    var expected = [
      {dest: 'expand/deep/deep.txt', src: ['expand/deep/deep.txt']},
      {dest: 'expand/deep/deeper/deeper.txt', src: ['expand/deep/deeper/deeper.txt']},
      {dest: 'expand/deep/deeper/deepest/deepest.txt', src: ['expand/deep/deeper/deepest/deepest.txt']},
    ];
    test.deepEqual(actual, expected, 'default options');

    expected = globule.mapping(globule.find(['expand/**/*.txt']));
    test.deepEqual(actual, expected, 'this is what it\'s doing under the hood, anwyays.');

    test.done();
  },
  'options.srcBase': function(test) {
    test.expect(1);
    var actual = globule.findMapping(['**/*.txt'], {destBase: 'dest', srcBase: 'expand/deep'});
    var expected = [
      {dest: 'dest/deep.txt', src: ['expand/deep/deep.txt']},
      {dest: 'dest/deeper/deeper.txt', src: ['expand/deep/deeper/deeper.txt']},
      {dest: 'dest/deeper/deepest/deepest.txt', src: ['expand/deep/deeper/deepest/deepest.txt']},
    ];
    test.deepEqual(actual, expected, 'srcBase should be stripped from front of destPath, pre-destBase+destPath join');
    test.done();
  },
};
