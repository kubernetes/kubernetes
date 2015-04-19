var replaceExt = require('../');
var path = require('path');
var should = require('should');
require('mocha');

describe('replace-ext', function() {
  it('should return a valid replaced extension on nested', function(done) {
    var fname = path.join(__dirname, './fixtures/test.coffee');
    var expected = path.join(__dirname, './fixtures/test.js');
    var nu = replaceExt(fname, '.js');
    should.exist(nu);
    nu.should.equal(expected);
    done();
  });

  it('should return a valid replaced extension on flat', function(done) {
    var fname = 'test.coffee';
    var expected = 'test.js';
    var nu = replaceExt(fname, '.js');
    should.exist(nu);
    nu.should.equal(expected);
    done();
  });

  it('should not return a valid replaced extension on empty string', function(done) {
    var fname = '';
    var expected = '';
    var nu = replaceExt(fname, '.js');
    should.exist(nu);
    nu.should.equal(expected);
    done();
  });

  it('should return a valid removed extension on nested', function(done) {
    var fname = path.join(__dirname, './fixtures/test.coffee');
    var expected = path.join(__dirname, './fixtures/test');
    var nu = replaceExt(fname, '');
    should.exist(nu);
    nu.should.equal(expected);
    done();
  });

  it('should return a valid added extension on nested', function(done) {
    var fname = path.join(__dirname, './fixtures/test');
    var expected = path.join(__dirname, './fixtures/test.js');
    var nu = replaceExt(fname, '.js');
    should.exist(nu);
    nu.should.equal(expected);
    done();
  });
});
