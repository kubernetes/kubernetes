var deprecated = require('../');
var should = require('should');
require('mocha');

describe('method()', function() {
  it('should return a wrapped function that logs once', function(done) {
    var message = 'testing';
    var scope = {
      a: 1
    };
    var logged = false;
    var log = function(msg){
      msg.should.equal(message);
      logged.should.equal(false);
      logged = true;
    };
    var fn = deprecated.method(message, log, function(one, two){
      this.should.equal(scope);
      one.should.equal(1);
      two.should.equal(2);
      return one+two;
    });

    fn.bind(scope)(1,2).should.equal(3);
    fn.bind(scope)(1,2).should.equal(3);
    fn.bind(scope)(1,2).should.equal(3);
    fn.bind(scope)(1,2).should.equal(3);

    logged.should.equal(true);
    done();
  });
});