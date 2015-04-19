var expect = require('chai').expect
  , unique = require('..')
  , Stream = require('stream')
  , after = require('after')
  , setImmediate = global.setImmediate || process.nextTick;

describe('unique stream', function() {

  function makeStream(type) {
    var s = new Stream();
    s.readable = true;

    var n = 10;
    var next = after(n, function () {
      setImmediate(function () {
        s.emit('end');
      });
    });

    for (var i = 0; i < n; i++) {
      var o = {
        type: type,
        name: 'name ' + i,
        number: i * 10
      };

      (function (o) {
        setImmediate(function () {
          s.emit('data', o);
          next();
        });
      })(o);
    }
    return s;
  }

  it('should be able to uniqueify objects based on JSON data', function(done) {
    var aggregator = unique();
    makeStream('a')
      .pipe(aggregator);
    makeStream('a')
      .pipe(aggregator);

    var n = 0;
    aggregator
      .on('data', function () {
        n++;
      })
      .on('end', function () {
        expect(n).to.equal(10);
        done();
      });
  });

  it('should be able to uniqueify objects based on a property', function(done) {
    var aggregator = unique('number');
    makeStream('a')
      .pipe(aggregator);
    makeStream('b')
      .pipe(aggregator);

    var n = 0;
    aggregator
      .on('data', function () {
        n++;
      })
      .on('end', function () {
        expect(n).to.equal(10);
        done();
      });
  });

  it('should be able to uniqueify objects based on a function', function(done) {
    var aggregator = unique(function (data) {
      return data.name;
    });

    makeStream('a')
      .pipe(aggregator);
    makeStream('b')
      .pipe(aggregator);

    var n = 0;
    aggregator
      .on('data', function () {
        n++;
      })
      .on('end', function () {
        expect(n).to.equal(10);
        done();
      });
  });

  it('should be able to handle uniqueness when not piped', function(done) {
    var stream = unique();
    var count = 0;
    stream.on('data', function (data) {
      expect(data).to.equal('hello');
      count++;
    });
    stream.on('end', function() {
      expect(count).to.equal(1);
      done();
    });
    stream.write('hello');
    stream.write('hello');
    stream.end();
  });
});
