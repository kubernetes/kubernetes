describe('a pass and a failure', function(){
  describe('pass', function() {
    it('should pass', function(){
      expect(1+2).toEqual(3);
    });
  });
  describe('failure', function(){
    it('should report failure (THIS IS EXPECTED)', function(){
      expect(true).toBeFalsy();
    });
  });
});

describe('Testing waitsfor functionality', function() {
    it('runs and then waitsFor should timeout (THIS IS EXPECTED)', function() {
        runs(function() {
            1+1;
        });
        waitsFor(function() {
            return true === false;
        }, 'the impossible', 1000);
        runs(function() {
            expect(true).toBeTruthy();
        });
    });
});

describe('timeouts', function() {
  jasmine.getEnv().defaultTimeoutInterval = 44;

  it('should timeout after 44ms (THIS IS EXPECTED)', function(done) {
    setTimeout(function() {
      expect(true).toBe(true);
      done();
    }, 1000);
  });

  it('should timeout after 55ms (THIS IS EXPECTED)', function(done) {
    setTimeout(function() {
      expect(true).toBe(true);
      done();
    }, 1000);
  }, 55);
});
