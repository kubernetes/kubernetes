var
  vows   = require('vows'),
  assert = require('assert'),

  path   = require('path'),

  tmp    = require('../lib/tmp.js'),
  Test   = require('./base.js');

vows.describe('Name creation').addBatch({
  'when using without parameters': {
    topic: function () {
      tmp.tmpName(this.callback);
    },

    'should not return with error': assert.isNull,
    'should have the default prefix': Test.testPrefix('tmp-')
  },

  'when using with prefix': {
    topic: function () {
      tmp.tmpName({ prefix: 'something' }, this.callback);
    },

    'should not return with error': assert.isNull,
    'should have the provided prefix': Test.testPrefix('something')
  },

  'when using with postfix': {
    topic: function () {
      tmp.tmpName({ postfix: '.txt' }, this.callback);
    },

    'should not return with error': assert.isNull,
    'should have the provided postfix': Test.testPostfix('.txt')

  },

  'when using template': {
    topic: function () {
      tmp.tmpName({ template: path.join(tmp.tmpdir, 'clike-XXXXXX-postfix') }, this.callback);
    },

    'should not return with error': assert.isNull,
    'should have the provided prefix': Test.testPrefix('clike-'),
    'should have the provided postfix': Test.testPostfix('-postfix'),
    'should have template filled': function (err, name) {
      assert.isTrue(/[a-zA-Z0-9]{6}/.test(name));
    }
  },

  'when using multiple options': {
    topic: function () {
      tmp.tmpName({ prefix: 'foo', postfix: 'bar', tries: 5 }, this.callback);
    },

    'should not return with error': assert.isNull,
    'should have the provided prefix': Test.testPrefix('foo'),
    'should have the provided postfix': Test.testPostfix('bar')
  },

  'no tries': {
    topic: function () {
      tmp.tmpName({ tries: -1 }, this.callback);
    },

    'should fail': function (err, name) {
      assert.isObject(err);
    }
  },

  'tries not numeric': {
    topic: function () {
      tmp.tmpName({ tries: 'hello'}, this.callback);
    },

    'should fail': function (err, name) {
      assert.isObject(err);
    }
  }

}).exportTo(module);
