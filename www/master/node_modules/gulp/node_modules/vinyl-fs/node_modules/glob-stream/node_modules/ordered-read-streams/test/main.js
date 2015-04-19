var should = require('should');
var through = require('through2');
var OrderedStreams = require('../');

describe('ordered-read-streams', function () {
  it('should end if no streams are given', function (done) {
    var streams = OrderedStreams();
    streams.on('data', function () {
      done('error');
    });
    streams.on('end', done);
  });

  it('should throw error if one or more streams are not readable', function (done) {
    var writable = { readable: false };

    try {
      new OrderedStreams(writable);
    } catch (e) {
      e.message.should.equal('All input streams must be readable');
      done();
    }
  });

  it('should emit data from all streams', function(done) {
    var s1 = through.obj(function (data, enc, next) {
      this.push(data);
      next();
    });
    var s2 = through.obj(function (data, enc, next) {
      this.push(data);
      next();
    });
    var s3 = through.obj(function (data, enc, next) {
      this.push(data);
      next();
    });

    var streams = new OrderedStreams([s1, s2, s3]);
    var results = [];
    streams.on('data', function (data) {
      results.push(data);
    });
    streams.on('end', function () {
      results.length.should.be.exactly(3);
      results[0].should.equal('stream 1');
      results[1].should.equal('stream 2');
      results[2].should.equal('stream 3');
      done();
    });

    s1.write('stream 1');
    s1.end();

    s2.write('stream 2');
    s2.end();

    s3.write('stream 3');
    s3.end();
  });

  it('should emit all data event from each stream', function (done) {
    var s = through.obj(function (data, enc, next) {
      this.push(data);
      next();
    });

    var streams = new OrderedStreams(s);
    var results = [];
    streams.on('data', function (data) {
      results.push(data);
    });
    streams.on('end', function () {
      results.length.should.be.exactly(3);
      done();
    });

    s.write('data1');
    s.write('data2');
    s.write('data3');
    s.end();
  });

  it('should preserve streams order', function(done) {
    var s1 = through.obj(function (data, enc, next) {
      var self = this;
      setTimeout(function () {
        self.push(data);
        next();
      }, 200);
    });
    var s2 = through.obj(function (data, enc, next) {
      var self = this;
      setTimeout(function () {
        self.push(data);
        next();
      }, 30);
    });
    var s3 = through.obj(function (data, enc, next) {
      var self = this;
      setTimeout(function () {
        self.push(data);
        next();
      }, 100);
    });

    var streams = new OrderedStreams([s1, s2, s3]);
    var results = [];
    streams.on('data', function (data) {
      results.push(data);
    });
    streams.on('end', function () {
      results.length.should.be.exactly(3);
      results[0].should.equal('stream 1');
      results[1].should.equal('stream 2');
      results[2].should.equal('stream 3');
      done();
    });

    s1.write('stream 1');
    s1.end();

    s2.write('stream 2');
    s2.end();

    s3.write('stream 3');
    s3.end();
  });

  it('should emit stream errors downstream', function (done) {
    var s = through.obj(function (data, enc, next) {
      this.emit('error', new Error('stahp!'));
      next();
    });
    var s2 = through.obj(function (data, enc, next) {
      this.push(data);
      next();
    });

    var errMsg;
    var streamData;
    var streams = new OrderedStreams([s, s2]);
    streams.on('data', function (data) {
      streamData = data;
    });
    streams.on('error', function (err) {
      errMsg = err.message;
    });
    streams.on('end', function () {
      errMsg.should.equal('stahp!');
      streamData.should.equal('Im ok!');
      done();
    });

    s.write('go');
    s.end();
    s2.write('Im ok!');
    s2.end();
  });
});
