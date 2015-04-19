var isNull = require('../lib/isNull');
var Stream = require('stream');
var should = require('should');
require('mocha');

describe('isNull()', function() {
  it('should return true on null', function(done) {
    isNull(null).should.equal(true);
    done();
  });

  it('should return false on undefined', function(done) {
    isNull().should.equal(false);
    isNull(undefined).should.equal(false);
    done();
  });

  it('should return false on defined values', function(done) {
    isNull(1).should.equal(false);
    isNull("test").should.equal(false);
    done();
  });
});