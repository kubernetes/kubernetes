var deprecated = require('../');
var should = require('should');
require('mocha');

describe('field()', function() {
  it('should return a wrapped function that logs once on get', function(done) {
    var message = 'testing';
    var scope = {
      a: 1
    };
    var obj = {};
    var logged = false;
    var log = function(msg){
      msg.should.equal(message);
      logged.should.equal(false);
      logged = true;
    };
    deprecated.field(message, log, obj, 'a', 123);

    obj.a.should.equal(123);
    obj.a = 1234;
    obj.a.should.equal(1234);
    logged.should.equal(true);
    done();
  });
  it('should return a wrapped function that logs once on set', function(done) {
    var message = 'testing';
    var scope = {
      a: 1
    };
    var obj = {};
    var logged = false;
    var log = function(msg){
      msg.should.equal(message);
      logged.should.equal(false);
      logged = true;
    };
    deprecated.field(message, log, obj, 'a', 123);

    obj.a = 1234;
    logged.should.equal(true);
    done();
  });
});