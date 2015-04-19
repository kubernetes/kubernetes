
describe('jasmine-node-flat', function(){
  it('should pass', function(){
    expect(1+2).toEqual(3);
  });
});

describe('Testing some characters', function()  {
    var chars = ['&', '\'', '"', '<', '>'];
    for(var i = 0; i < chars.length; i+=1)  {
        currentChar = chars[i];
        it('should reject ' + currentChar, (function(currentChar)  {
            expect(false).toEqual(false);
        })(currentChar));
    }
});

describe('root', function () {

  describe('nested', function () {

    xit('nested statement', function () {
      expect(1).toBeTruthy();
    });

  });

  it('root statement', function () {
    expect(1).toBeTruthy();
  });

});

describe("Top level describe block", function() {
  it("first it block in top level describe", function() {
    expect(true).toEqual(true);
  });
  describe("Second level describe block", function() {
    it("first it block in second level describe", function() {
      expect(true).toBe(true);
    });
  });
  it("second it block in top level describe", function() {
    expect(true).toEqual(true);
  });
});

describe('async', function () {

    var request = function (str, func) {
        func('1', '2', 'hello world');
    };

    it("should respond with hello world", function(done) {
        request("http://localhost:3000/hello", function(error, response, body){
            expect(body).toEqual("hello world");
            done();
        });
    });

    it("should respond with hello world", function(done) {
        request("http://localhost:3000/hello", function(error, response, body){
            expect(body).toEqual("hello world");
            done();
        });
    }, 250); // timeout after 250 ms

});
