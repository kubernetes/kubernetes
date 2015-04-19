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

var AdmZip = require('adm-zip'),
    assert = require('assert'),
    fs = require('fs'),
    path = require('path');

var promise = require('../..').promise,
    Profile = require('../../firefox/profile').Profile,
    decode = require('../../firefox/profile').decode,
    loadUserPrefs = require('../../firefox/profile').loadUserPrefs,
    io = require('../../io'),
    it = require('../../testing').it;


var JETPACK_EXTENSION = path.join(__dirname,
    '../../lib/test/data/firefox/jetpack-sample.xpi');
var NORMAL_EXTENSION = path.join(__dirname,
    '../../lib/test/data/firefox/sample.xpi');

var JETPACK_EXTENSION_ID = 'jid1-EaXX7k0wwiZR7w@jetpack.xpi';
var NORMAL_EXTENSION_ID = 'sample@seleniumhq.org';
var WEBDRIVER_EXTENSION_ID = 'fxdriver@googlecode.com';



describe('Profile', function() {
  describe('setPreference', function() {
    it('allows setting custom properties', function() {
      var profile = new Profile();
      assert.equal(undefined, profile.getPreference('foo'));

      profile.setPreference('foo', 'bar');
      assert.equal('bar', profile.getPreference('foo'));
    });

    it('allows overriding mutable properties', function() {
      var profile = new Profile();
      assert.equal('about:blank', profile.getPreference('browser.newtab.url'));

      profile.setPreference('browser.newtab.url', 'http://www.example.com');
      assert.equal('http://www.example.com',
          profile.getPreference('browser.newtab.url'));
    });

    it('throws if setting a frozen preference', function() {
      var profile = new Profile();
      assert.throws(function() {
        profile.setPreference('app.update.auto', true);
      });
    });
  });

  describe('writeToDisk', function() {
    it('copies template directory recursively', function() {
      var templateDir;
      return io.tmpDir().then(function(td) {
        templateDir = td;
        var foo = path.join(templateDir, 'foo');
        fs.writeFileSync(foo, 'Hello, world');

        var bar = path.join(templateDir, 'subfolder/bar');
        fs.mkdirSync(path.dirname(bar));
        fs.writeFileSync(bar, 'Goodbye, world!');

        return new Profile(templateDir).writeToDisk();
      }).then(function(profileDir) {
        assert.notEqual(profileDir, templateDir);

        assert.equal('Hello, world',
            fs.readFileSync(path.join(profileDir, 'foo')));
        assert.equal('Goodbye, world!',
            fs.readFileSync(path.join(profileDir, 'subfolder/bar')));
      });
    });

    it('does not copy lock files', function() {
      return io.tmpDir().then(function(dir) {
        fs.writeFileSync(path.join(dir, 'parent.lock'), 'lock');
        fs.writeFileSync(path.join(dir, 'lock'), 'lock');
        fs.writeFileSync(path.join(dir, '.parentlock'), 'lock');
        return new Profile(dir).writeToDisk();
      }).then(function(dir) {
        assert.ok(fs.existsSync(dir));
        assert.ok(!fs.existsSync(path.join(dir, 'parent.lock')));
        assert.ok(!fs.existsSync(path.join(dir, 'lock')));
        assert.ok(!fs.existsSync(path.join(dir, '.parentlock')));
      });
    });

    describe('user.js', function() {

      it('writes defaults', function() {
        return new Profile().writeToDisk().then(function(dir) {
          return loadUserPrefs(path.join(dir, 'user.js'));
        }).then(function(prefs) {
          // Just check a few.
          assert.equal(false, prefs['app.update.auto']);
          assert.equal(true, prefs['browser.EULA.override']);
          assert.equal(false, prefs['extensions.update.enabled']);
          assert.equal('about:blank', prefs['browser.newtab.url']);
          assert.equal(30, prefs['dom.max_script_run_time']);
        });
      });

      it('merges template user.js into preferences', function() {
        return io.tmpDir().then(function(dir) {
          fs.writeFileSync(path.join(dir, 'user.js'), [
            'user_pref("browser.newtab.url", "http://www.example.com")',
            'user_pref("dom.max_script_run_time", 1234)'
          ].join('\n'));

          return new Profile(dir).writeToDisk();
        }).then(function(profile) {
          return loadUserPrefs(path.join(profile, 'user.js'));
        }).then(function(prefs) {
          assert.equal('http://www.example.com', prefs['browser.newtab.url']);
          assert.equal(1234, prefs['dom.max_script_run_time']);
        });
      });

      it('ignores frozen preferences when merging template user.js',
        function() {
          return io.tmpDir().then(function(dir) {
            fs.writeFileSync(path.join(dir, 'user.js'),
                'user_pref("app.update.auto", true)');
            return new Profile(dir).writeToDisk();
          }).then(function(profile) {
            return loadUserPrefs(path.join(profile, 'user.js'));
          }).then(function(prefs) {
            assert.equal(false, prefs['app.update.auto']);
          });
        });
    });

    describe('extensions', function() {
      it('are copied into new profile directory', function() {
        var profile = new Profile();
        profile.addExtension(JETPACK_EXTENSION);
        profile.addExtension(NORMAL_EXTENSION);

        return profile.writeToDisk().then(function(dir) {
          dir = path.join(dir, 'extensions');
          assert.ok(fs.existsSync(path.join(dir, JETPACK_EXTENSION_ID)));
          assert.ok(fs.existsSync(path.join(dir, NORMAL_EXTENSION_ID)));
          assert.ok(fs.existsSync(path.join(dir, WEBDRIVER_EXTENSION_ID)));
        });
      });
    });
  });

  describe('encode', function() {
    it('excludes the bundled WebDriver extension', function() {
      return new Profile().encode().then(function(data) {
        return decode(data);
      }).then(function(dir) {
        assert.ok(fs.existsSync(path.join(dir, 'user.js')));
        assert.ok(fs.existsSync(path.join(dir, 'extensions')));
        return loadUserPrefs(path.join(dir, 'user.js'));
      }).then(function(prefs) {
        // Just check a few.
        assert.equal(false, prefs['app.update.auto']);
        assert.equal(true, prefs['browser.EULA.override']);
        assert.equal(false, prefs['extensions.update.enabled']);
        assert.equal('about:blank', prefs['browser.newtab.url']);
        assert.equal(30, prefs['dom.max_script_run_time']);
      });
    });
  });
});
