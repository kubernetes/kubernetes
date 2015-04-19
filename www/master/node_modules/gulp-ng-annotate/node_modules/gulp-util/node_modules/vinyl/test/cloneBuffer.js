var cloneBuffer = require('../lib/cloneBuffer');
var Stream = require('stream');
var should = require('should');
require('mocha');

describe('cloneBuffer()', function() {
  it('should return a new Buffer reference', function(done) {
    var testBuffer = new Buffer('test');
    var testBuffer2 = cloneBuffer(testBuffer);

    should.exist(testBuffer2, 'should return something');
    (testBuffer2 instanceof Buffer).should.equal(true, 'should return a Buffer');
    testBuffer2.should.not.equal(testBuffer, 'pointer should change');
    done();
  });

  it('should not replicate modifications to the original buffer', function(done) {
    var testBuffer = new Buffer('test');
    var testBuffer2 = cloneBuffer(testBuffer);

    // test that changes dont modify both pointers
    testBuffer2.write('w');

    testBuffer.toString('utf8').should.equal('test', 'original should stay the same');
    testBuffer2.toString('utf8').should.equal('west', 'new buffer should be modified');
    done();
  });
});