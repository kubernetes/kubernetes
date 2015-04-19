var
  vows   = require('vows'),
  assert = require('assert'),

  path       = require('path'),
  fs         = require('fs'),
  existsSync = fs.existsSync || path.existsSync,

  tmp    = require('../lib/tmp.js'),
  Test   = require('./base.js');


function _testDir(mode) {
  return function _testDirGenerated(err, name) {
    assert.ok(existsSync(name), 'should exist');

    var stat = fs.statSync(name);
    assert.ok(stat.isDirectory(), 'should be a directory');

    Test.testStat(stat, mode);
  };
}

vows.describe('Directory creation').addBatch({
  'when using without parameters': {
    topic: function () {
      tmp.dir(this.callback);
    },

    'should be a directory': _testDir(040700),
    'should have the default prefix': Test.testPrefix('tmp-')
  },

  'when using with prefix': {
    topic: function () {
      tmp.dir({ prefix: 'something' }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a directory': _testDir(040700),
    'should have the provided prefix': Test.testPrefix('something')
  },

  'when using with postfix': {
    topic: function () {
      tmp.dir({ postfix: '.txt' }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a directory': _testDir(040700),
    'should have the provided postfix': Test.testPostfix('.txt')
  },

  'when using template': {
    topic: function () {
      tmp.dir({ template: path.join(tmp.tmpdir, 'clike-XXXXXX-postfix') }, this.callback);
    },

    'should not return with error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': _testDir(040700),
    'should have the provided prefix': Test.testPrefix('clike-'),
    'should have the provided postfix': Test.testPostfix('-postfix')
  },

  'when using multiple options': {
    topic: function () {
      tmp.dir({ prefix: 'foo', postfix: 'bar', mode: 0750 }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a directory': _testDir(040750),
    'should have the provided prefix': Test.testPrefix('foo'),
    'should have the provided postfix': Test.testPostfix('bar')
  },

  'when using multiple options and mode': {
    topic: function () {
      tmp.dir({ prefix: 'complicated', postfix: 'options', mode: 0755 }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a directory': _testDir(040755),
    'should have the provided prefix': Test.testPrefix('complicated'),
    'should have the provided postfix': Test.testPostfix('options')
  },

  'no tries': {
    topic: function () {
      tmp.dir({ tries: -1 }, this.callback);
    },

    'should return with an error': assert.isObject
  },

  'keep testing': {
    topic: function () {
      Test.testKeep('dir', '1', this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a dir': function (err, name) {
      _testDir(040700)(err, name);
      fs.rmdirSync(name);
    }
  },

  'unlink testing': {
    topic: function () {
      Test.testKeep('dir', '0', this.callback);
    },

    'should not return with error': assert.isNull,
    'should return with a name': Test.assertName,
    'should not exist': function (err, name) {
      assert.ok(!existsSync(name), "Directory should be removed");
    }
  },

  'non graceful testing': {
    topic: function () {
      Test.testGraceful('dir', '0', this.callback);
    },

    'should not return with error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a dir': function (err, name) {
      _testDir(040700)(err, name);
      fs.rmdirSync(name);
    }
  },

  'graceful testing': {
    topic: function () {
      Test.testGraceful('dir', '1', this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should not exist': function (err, name) {
      assert.ok(!existsSync(name), "Directory should be removed");
    }
  },

  'unsafeCleanup === true': {
    topic: function () {
      Test.testUnsafeCleanup('1', this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should not exist': function (err, name) {
      assert.ok(!existsSync(name), "Directory should be removed");
    },
    'should remove symlinked dir': function(err, name) {
      assert.ok(
        !existsSync(name + '/symlinkme-target'),
        'should remove target'
      );
    },
    'should not remove contents of symlink dir': function(err, name) {
      assert.ok(
        existsSync(__dirname + '/symlinkme/file.js'),
        'should not remove symlinked directory\'s content'
      );
    }
  },

  'unsafeCleanup === false': {
    topic: function () {
      Test.testUnsafeCleanup('0', this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a directory': _testDir(040700)
  },

  'remove callback': {
    topic: function () {
      tmp.dir(this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'removeCallback should remove directory': function (_err, name, removeCallback) {
      removeCallback();
      assert.ok(!existsSync(name), "Directory should be removed");
    }
  }
}).exportTo(module);
