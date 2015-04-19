// Copyright 2014 Selenium committers
// Copyright 2014 Software Freedom Conservancy
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

var assert = require('assert'),
    fs = require('fs'),
    path = require('path'),
    tmp = require('tmp');

var io = require('../io'),
    before = require('../testing').before,
    beforeEach = require('../testing').beforeEach,
    it = require('../testing').it;


describe('io', function() {
  describe('copy', function() {
    var tmpDir;

    before(function() {
      return io.tmpDir().then(function(d) {
        tmpDir = d;

        fs.writeFileSync(path.join(d, 'foo'), 'Hello, world');
        fs.symlinkSync(path.join(d, 'foo'), path.join(d, 'symlinked-foo'));
      });
    });

    it('can copy one file to another', function() {
      return io.tmpFile().then(function(f) {
        return io.copy(path.join(tmpDir, 'foo'), f).then(function(p) {
          assert.equal(p, f);
          assert.equal('Hello, world', fs.readFileSync(p));
        });
      });
    });

    it('can copy symlink to destination', function() {
      return io.tmpFile().then(function(f) {
        return io.copy(path.join(tmpDir, 'symlinked-foo'), f).then(function(p) {
          assert.equal(p, f);
          assert.equal('Hello, world', fs.readFileSync(p));
        });
      });
    });

    it('fails if given a directory as a source', function() {
      return io.tmpFile().then(function(f) {
        return io.copy(tmpDir, f);
      }).then(function() {
        throw Error('Should have failed with a type error');
      }, function() {
        // Do nothing; expected.
      });
    });
  });

  describe('copyDir', function() {
    it('copies recursively', function() {
      return io.tmpDir().then(function(dir) {
        fs.writeFileSync(path.join(dir, 'file1'), 'hello');
        fs.mkdirSync(path.join(dir, 'sub'));
        fs.mkdirSync(path.join(dir, 'sub/folder'));
        fs.writeFileSync(path.join(dir, 'sub/folder/file2'), 'goodbye');

        return io.tmpDir().then(function(dst) {
          return io.copyDir(dir, dst).then(function(ret) {
            assert.equal(dst, ret);

            assert.equal('hello',
              fs.readFileSync(path.join(dst, 'file1')));
            assert.equal('goodbye',
              fs.readFileSync(path.join(dst, 'sub/folder/file2')));
          });
        });
      });
    });

    it('creates destination dir if necessary', function() {
      return io.tmpDir().then(function(srcDir) {
        fs.writeFileSync(path.join(srcDir, 'foo'), 'hi');
        return io.tmpDir().then(function(dstDir) {
          return io.copyDir(srcDir, path.join(dstDir, 'sub'));
        });
      }).then(function(p) {
        assert.equal('sub', path.basename(p));
        assert.equal('hi', fs.readFileSync(path.join(p, 'foo')));
      });
    });

    it('supports regex exclusion filter', function() {
      return io.tmpDir().then(function(src) {
        fs.writeFileSync(path.join(src, 'foo'), 'a');
        fs.writeFileSync(path.join(src, 'bar'), 'b');
        fs.writeFileSync(path.join(src, 'baz'), 'c');
        fs.mkdirSync(path.join(src, 'sub'));
        fs.writeFileSync(path.join(src, 'sub/quux'), 'd');
        fs.writeFileSync(path.join(src, 'sub/quot'), 'e');

        return io.tmpDir().then(function(dst) {
          return io.copyDir(src, dst, /(bar|quux)/);
        });
      }).then(function(dir) {
        assert.equal('a', fs.readFileSync(path.join(dir, 'foo')));
        assert.equal('c', fs.readFileSync(path.join(dir, 'baz')));
        assert.equal('e', fs.readFileSync(path.join(dir, 'sub/quot')));

        assert.ok(!fs.existsSync(path.join(dir, 'bar')));
        assert.ok(!fs.existsSync(path.join(dir, 'sub/quux')));
      });
    });

    it('supports exclusion filter function', function() {
      return io.tmpDir().then(function(src) {
        fs.writeFileSync(path.join(src, 'foo'), 'a');
        fs.writeFileSync(path.join(src, 'bar'), 'b');
        fs.writeFileSync(path.join(src, 'baz'), 'c');
        fs.mkdirSync(path.join(src, 'sub'));
        fs.writeFileSync(path.join(src, 'sub/quux'), 'd');
        fs.writeFileSync(path.join(src, 'sub/quot'), 'e');

        return io.tmpDir().then(function(dst) {
          return io.copyDir(src, dst, function(f) {
            return f !== path.join(src, 'foo')
                && f !== path.join(src, 'sub/quot');
          });
        });
      }).then(function(dir) {
        assert.equal('b', fs.readFileSync(path.join(dir, 'bar')));
        assert.equal('c', fs.readFileSync(path.join(dir, 'baz')));
        assert.equal('d', fs.readFileSync(path.join(dir, 'sub/quux')));

        assert.ok(!fs.existsSync(path.join(dir, 'foo')));
        assert.ok(!fs.existsSync(path.join(dir, 'sub/quot')));
      });
    });
  });

  describe('exists', function() {
    var dir;

    before(function() {
      return io.tmpDir().then(function(d) {
        dir = d;
      });
    });

    it('works for directories', function() {
      return io.exists(dir).then(assert.ok);
    });

    it('works for files', function() {
      var file = path.join(dir, 'foo');
      fs.writeFileSync(file, '');
      return io.exists(file).then(assert.ok);
    });

    it('does not return a rejected promise if file does not exist', function() {
      return io.exists(path.join(dir, 'not-there')).then(function(exists) {
        assert.ok(!exists);
      });
    });
  });
});
