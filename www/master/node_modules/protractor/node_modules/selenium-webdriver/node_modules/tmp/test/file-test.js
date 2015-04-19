var
  vows   = require('vows'),
  assert = require('assert'),

  path       = require('path'),
  fs         = require('fs'),
  existsSync = fs.existsSync || path.existsSync,

  tmp    = require('../lib/tmp.js'),
  Test   = require('./base.js');


function _testFile(mode, fdTest) {
  return function _testFileGenerated(err, name, fd) {
    assert.ok(existsSync(name), 'should exist');

    var stat = fs.statSync(name);
    assert.equal(stat.size, 0, 'should have zero size');
    assert.ok(stat.isFile(), 'should be a file');

    Test.testStat(stat, mode);

    // check with fstat as well (fd checking)
    if (fdTest) {
      var fstat = fs.fstatSync(fd);
      assert.deepEqual(fstat, stat, 'fstat results should be the same');

      var data = new Buffer('something');
      assert.equal(fs.writeSync(fd, data, 0, data.length, 0), data.length, 'should be writable');
      assert.ok(!fs.closeSync(fd), 'should not return with error');
    }
  };
}

vows.describe('File creation').addBatch({
  'when using without parameters': {
    topic: function () {
      tmp.file(this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': _testFile(0100600, true),
    'should have the default prefix': Test.testPrefix('tmp-'),
    'should have the default postfix': Test.testPostfix('.tmp')
  },

  'when using with prefix': {
    topic: function () {
      tmp.file({ prefix: 'something' }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': _testFile(0100600, true),
    'should have the provided prefix': Test.testPrefix('something')
  },

  'when using with postfix': {
    topic: function () {
      tmp.file({ postfix: '.txt' }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': _testFile(0100600, true),
    'should have the provided postfix': Test.testPostfix('.txt')
  },

  'when using template': {
    topic: function () {
      tmp.file({ template: path.join(tmp.tmpdir, 'clike-XXXXXX-postfix') }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': _testFile(0100600, true),
    'should have the provided prefix': Test.testPrefix('clike-'),
    'should have the provided postfix': Test.testPostfix('-postfix')
  },

  'when using multiple options': {
    topic: function () {
      tmp.file({ prefix: 'foo', postfix: 'bar', mode: 0640 }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': _testFile(0100640, true),
    'should have the provided prefix': Test.testPrefix('foo'),
    'should have the provided postfix': Test.testPostfix('bar')
  },

  'when using multiple options and mode': {
    topic: function () {
      tmp.file({ prefix: 'complicated', postfix: 'options', mode: 0644 }, this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': _testFile(0100644, true),
    'should have the provided prefix': Test.testPrefix('complicated'),
    'should have the provided postfix': Test.testPostfix('options')
  },

  'no tries': {
    topic: function () {
      tmp.file({ tries: -1 }, this.callback);
    },

    'should not be created': assert.isObject
  },

  'keep testing': {
    topic: function () {
      Test.testKeep('file', '1', this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': function (err, name) {
      _testFile(0100600, false)(err, name, null);
      fs.unlinkSync(name);
    }
  },

  'unlink testing': {
    topic: function () {
      Test.testKeep('file', '0', this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should not exist': function (err, name) {
      assert.ok(!existsSync(name), "File should be removed");
    }
  },

  'non graceful testing': {
    topic: function () {
      Test.testGraceful('file', '0', this.callback);
    },

    'should not return with error': assert.isNull,
    'should return with a name': Test.assertName,
    'should be a file': function (err, name) {
      _testFile(0100600, false)(err, name, null);
      fs.unlinkSync(name);
    }
  },

  'graceful testing': {
    topic: function () {
      Test.testGraceful('file', '1', this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'should not exist': function (err, name) {
      assert.ok(!existsSync(name), "File should be removed");
    }
  },

  'remove callback': {
    topic: function () {
      tmp.file(this.callback);
    },

    'should not return with an error': assert.isNull,
    'should return with a name': Test.assertName,
    'removeCallback should remove file': function (_err, name, _fd, removeCallback) {
      removeCallback();
      assert.ok(!existsSync(name), "File should be removed");
    }
  }

}).exportTo(module);
