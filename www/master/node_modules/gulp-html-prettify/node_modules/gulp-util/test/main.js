var util = require('../');
var should = require('should');
var join = require('path').join;
require('mocha');

describe('gulp-util', function() {
  describe('realBase()', function() {
    it('should return a valid shortened name', function(done) {
      var fname = join(__dirname, "./fixtures/test.coffee");
      var dname = join(__dirname, "./fixtures/");
      var shortened = util.realBase(dname, fname);
      should.exist(shortened);
      shortened.should.equal("test.coffee");
      done();
    });
  });

  describe('replaceExtension()', function() {
    it('should return a valid replaced extension on nested', function(done) {
      var fname = join(__dirname, "./fixtures/test.coffee");
      var expected = join(__dirname, "./fixtures/test.js");
      var nu = util.replaceExtension(fname, ".js");
      should.exist(nu);
      nu.should.equal(expected);
      done();
    });

    it('should return a valid replaced extension on flat', function(done) {
      var fname = "test.coffee";
      var expected = "test.js";
      var nu = util.replaceExtension(fname, ".js");
      should.exist(nu);
      nu.should.equal(expected);
      done();
    });

  });
});
