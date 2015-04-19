var assert = require('assert')
  , es = require('event-stream')
  , BufferStream = require('../src')
;

// Helpers
function syncBufferPrefixer(headerText) {
  return new BufferStream(function(err, buf, cb) {
    assert.equal(err, null);
    if(null === buf) {
      cb(null, Buffer(headerText));
    } else {
      cb(null, Buffer.concat([Buffer(headerText), buf]));
    }
  });
}
function asyncBufferPrefixer(headerText) {
  return new BufferStream(function(err, buf, cb) {
    assert.equal(err, null);
    if(null === buf) {
      setTimeout(function() {
        cb(null, Buffer(headerText));
      }, 0);
    } else {
      setTimeout(function() {
        cb(null, Buffer.concat([Buffer(headerText), buf]));
      }, 0);
    }
  });
}

// Tests
describe('Abstract buffers', function() {
  describe('synchonously', function() {

    it('should work with one pipe', function(done) {
      es.readArray(['te', 'st'])
        .pipe(syncBufferPrefixer('plop'))
        .pipe(es.wait(function(err, data) {
          assert.equal(err, null);
          assert.equal(data, 'ploptest');
          done();
        }));
    });

    it('should work when returning a null buffer', function(done) {
    
      es.readArray(['te', 'st'])
        .pipe(new BufferStream(function(err, buf, cb){
        cb(null, null);
        }))
        .pipe(es.wait(function(err, data) {
          assert.equal(err, null);
          assert.equal(data, '');
          done();
        }));
    });

    it('should work with multiple pipes', function(done) {
      es.readArray(['te', 'st'])
        .pipe(syncBufferPrefixer('plop'))
        .pipe(syncBufferPrefixer('plip'))
        .pipe(syncBufferPrefixer('plap'))
        .pipe(es.wait(function(err, data) {
          assert.equal(err, null);
          assert.equal(data, 'plapplipploptest');
          done();
        }));
    });

  });

  describe('asynchonously', function() {

    it('should work with one pipe', function(done) {
      es.readArray(['te', 'st'])
        .pipe(asyncBufferPrefixer('plop'))
        .pipe(es.wait(function(err, data) {
          assert.equal(err, null);
          assert.equal(data, 'ploptest');
          done();
        }));
    });

    it('should work when returning a null buffer', function(done) {
    
      es.readArray(['te', 'st'])
        .pipe(BufferStream(function(err, buf, cb){
        cb(null, null);
        }))
        .pipe(es.wait(function(err, data) {
          assert.equal(err, null);
          assert.equal(data, '');
          done();
        }));
    });

    it('should work with multiple pipes', function(done) {
      es.readArray(['te', 'st'])
        .pipe(asyncBufferPrefixer('plop'))
        .pipe(asyncBufferPrefixer('plip'))
        .pipe(asyncBufferPrefixer('plap'))
        .pipe(es.wait(function(err, data) {
          assert.equal(err, null);
          assert.equal(data, 'plapplipploptest');
          done();
        }));
    });

  });
});
