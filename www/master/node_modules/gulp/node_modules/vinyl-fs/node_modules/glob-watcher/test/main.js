var watch = require('../');
var should = require('should');
var path = require('path');
var fs = require('fs');
var rimraf = require('rimraf');
var mkdirp = require('mkdirp');

require('mocha');

describe('glob-watcher', function() {
  it('should return a valid file struct via EE', function(done) {
    var expectedName = path.join(__dirname, "./fixtures/stuff/temp.coffee");
    var fname = path.join(__dirname, "./fixtures/**/temp.coffee");
    mkdirp.sync(path.dirname(expectedName));
    fs.writeFileSync(expectedName, "testing");

    var watcher = watch(fname);
    watcher.on('change', function(evt) {
      should.exist(evt);
      should.exist(evt.path);
      should.exist(evt.type);
      evt.type.should.equal('changed');
      evt.path.should.equal(expectedName);
      watcher.end();
    });
    watcher.on('end', function(){
      rimraf.sync(expectedName);
      done();
    });
    setTimeout(function(){
      fs.writeFileSync(expectedName, "test test");
    }, 125);
  });

  it('should emit nomatch via EE', function(done) {
    var fname = path.join(__dirname, "./doesnt_exist_lol/temp.coffee");

    var watcher = watch(fname);
    watcher.on('nomatch', function() {
      done();
    });
  });

  it('should return a valid file struct via callback', function(done) {
    var expectedName = path.join(__dirname, "./fixtures/stuff/test.coffee");
    var fname = path.join(__dirname, "./fixtures/**/test.coffee");
    mkdirp.sync(path.dirname(expectedName));
    fs.writeFileSync(expectedName, "testing");

    var watcher = watch(fname, function(evt) {
      should.exist(evt);
      should.exist(evt.path);
      should.exist(evt.type);
      evt.type.should.equal('changed');
      evt.path.should.equal(expectedName);
      watcher.end();
    });

    watcher.on('end', function(){
      rimraf.sync(expectedName);
      done();
    });
    setTimeout(function(){
      fs.writeFileSync(expectedName, "test test");
    }, 200);
  });

  it('should not return a non-matching file struct via callback', function(done) {
    var expectedName = path.join(__dirname, "./fixtures/test123.coffee");
    var fname = path.join(__dirname, "./fixtures/**/test.coffee");
    mkdirp.sync(path.dirname(expectedName));
    fs.writeFileSync(expectedName, "testing");

    var watcher = watch(fname, function(evt) {
      throw new Error("Should not have been called! "+evt.path);
    });

    setTimeout(function(){
      fs.writeFileSync(expectedName, "test test");
    }, 200);

    setTimeout(function(){
      rimraf.sync(expectedName);
      done();
    }, 1500);
  });
});