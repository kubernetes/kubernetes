var assert = require("chai").assert;

var stream = require("readable-stream");

var duplexer2 = require("../");

describe("duplexer2", function() {
  var writable, readable;

  beforeEach(function() {
    writable = new stream.Writable({objectMode: true});
    readable = new stream.Readable({objectMode: true});

    writable._write = function _write(input, encoding, done) {
      return done();
    };

    readable._read = function _read(n) {
    };
  });

  it("should interact with the writable stream properly for writing", function(done) {
    var duplex = duplexer2(writable, readable);

    writable._write = function _write(input, encoding, _done) {
      assert.strictEqual(input, "well hello there");

      return done();
    };

    duplex.write("well hello there");
  });

  it("should interact with the readable stream properly for reading", function(done) {
    var duplex = duplexer2(writable, readable);

    duplex.on("data", function(e) {
      assert.strictEqual(e, "well hello there");

      return done();
    });

    readable.push("well hello there");
  });

  it("should end the writable stream, causing it to finish", function(done) {
    var duplex = duplexer2(writable, readable);

    writable.once("finish", done);

    duplex.end();
  });

  it("should finish when the writable stream finishes", function(done) {
    var duplex = duplexer2(writable, readable);

    duplex.once("finish", done);

    writable.end();
  });

  it("should end when the readable stream ends", function(done) {
    var duplex = duplexer2(writable, readable);

    // required to let "end" fire without reading
    duplex.resume();
    duplex.once("end", done);

    readable.push(null);
  });

  it("should bubble errors from the writable stream when no behaviour is specified", function(done) {
    var duplex = duplexer2(writable, readable);

    var originalErr = Error("testing");

    duplex.on("error", function(err) {
      assert.strictEqual(err, originalErr);

      return done();
    });

    writable.emit("error", originalErr);
  });

  it("should bubble errors from the readable stream when no behaviour is specified", function(done) {
    var duplex = duplexer2(writable, readable);

    var originalErr = Error("testing");

    duplex.on("error", function(err) {
      assert.strictEqual(err, originalErr);

      return done();
    });

    readable.emit("error", originalErr);
  });

  it("should bubble errors from the writable stream when bubbleErrors is true", function(done) {
    var duplex = duplexer2({bubbleErrors: true}, writable, readable);

    var originalErr = Error("testing");

    duplex.on("error", function(err) {
      assert.strictEqual(err, originalErr);

      return done();
    });

    writable.emit("error", originalErr);
  });

  it("should bubble errors from the readable stream when bubbleErrors is true", function(done) {
    var duplex = duplexer2({bubbleErrors: true}, writable, readable);

    var originalErr = Error("testing");

    duplex.on("error", function(err) {
      assert.strictEqual(err, originalErr);

      return done();
    });

    readable.emit("error", originalErr);
  });

  it("should not bubble errors from the writable stream when bubbleErrors is false", function(done) {
    var duplex = duplexer2({bubbleErrors: false}, writable, readable);

    var timeout = setTimeout(done, 25);

    duplex.on("error", function(err) {
      clearTimeout(timeout);

      return done(Error("shouldn't bubble error"));
    });

    // prevent uncaught error exception
    writable.on("error", function() {});

    writable.emit("error", Error("testing"));
  });

  it("should not bubble errors from the readable stream when bubbleErrors is false", function(done) {
    var duplex = duplexer2({bubbleErrors: false}, writable, readable);

    var timeout = setTimeout(done, 25);

    duplex.on("error", function(err) {
      clearTimeout(timeout);

      return done(Error("shouldn't bubble error"));
    });

    // prevent uncaught error exception
    readable.on("error", function() {});

    readable.emit("error", Error("testing"));
  });
});
