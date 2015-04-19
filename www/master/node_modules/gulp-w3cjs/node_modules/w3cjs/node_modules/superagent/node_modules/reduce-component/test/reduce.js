
var reduce = require('..');

describe('reduce', function(){

  describe('when adding prev and current', function(){
    it('should be sum all the values', function(){
      var numbers = [2,2,2];
      var fn = function(prev, curr){
        return prev + curr;
      };

      var a = numbers.reduce(fn);
      var b = reduce(numbers, fn);

      a.should.equal(6);
      b.should.equal(a);
    });
  });

  describe('when passing in an initial value', function(){
    it('should default to it', function(){
      var items = [];
      var fn = function(prev){
        return prev;
      };

      var a = items.reduce(fn, 'foo');
      var b = reduce(items, fn, 'foo');

      a.should.equal('foo');
      b.should.equal(a);
    });

    it('should start with it', function(){
      var items = [10, 10];
      var fn = function(prev, curr){
        return prev + curr;
      };

      var a = items.reduce(fn, 10);
      var b = reduce(items, fn, 10);

      a.should.equal(30);
      b.should.equal(a);
    });
  });

});