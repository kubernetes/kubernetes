var isBuffer = require('../lib/isBuffer');
var Stream = require('stream');
var should = require('should');
require('mocha');

describe('isBuffer()', function() {
  it('should return true on a Buffer', function(done) {
    var testBuffer = new Buffer('test');
    isBuffer(testBuffer).should.equal(true);
    done();
  });

  it('should return false on a Stream', function(done) {
    var testStream = new Stream();
    isBuffer(testStream).should.equal(false);
    done();
  });

  it('should return false on a null', function(done) {
    isBuffer(null).should.equal(false);
    done();
  });

  it('should return false on a array of numbers', function(done) {
    var testArray = [1, 2, 3];
    isBuffer(testArray).should.equal(false);
    done();
  });
});