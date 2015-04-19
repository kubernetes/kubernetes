var inspectStream = require('../lib/inspectStream');
var Stream = require('stream');
var should = require('should');
require('mocha');

describe('inspectStream()', function() {
  it('should work on a core Stream', function(done) {
    var testStream = new Stream();
    inspectStream(testStream).should.equal('<Stream>');
    done();
  });

  it('should work on a core Readable Stream', function(done) {
    var testStream = new Stream.Readable();
    inspectStream(testStream).should.equal('<ReadableStream>');
    done();
  });

  it('should work on a core Writable Stream', function(done) {
    var testStream = new Stream.Writable();
    inspectStream(testStream).should.equal('<WritableStream>');
    done();
  });

  it('should work on a core Duplex Stream', function(done) {
    var testStream = new Stream.Duplex();
    inspectStream(testStream).should.equal('<DuplexStream>');
    done();
  });

  it('should work on a core Transform Stream', function(done) {
    var testStream = new Stream.Transform();
    inspectStream(testStream).should.equal('<TransformStream>');
    done();
  });

  it('should work on a core PassThrough Stream', function(done) {
    var testStream = new Stream.PassThrough();
    inspectStream(testStream).should.equal('<PassThroughStream>');
    done();
  });

  it('should not work on a Buffer', function(done) {
    var testBuffer = new Buffer('test');
    should.not.exist(inspectStream(testBuffer));
    done();
  });

  it('should not work on a null', function(done) {
    should.not.exist(inspectStream(null));
    done();
  });
});