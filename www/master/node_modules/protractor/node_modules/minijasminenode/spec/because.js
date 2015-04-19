describe('because', function() {
  // test every functon that sets .message,
  // and ensure that .not still works....
  it('fails with 1 error, \'second:Expected \'blue\' to be \'red.\'',
      function() {
    var expected = 'blue';
    because('first').expect(expected).toBe('blue');
    because('second').expect(expected).toBe('red');
  });

  it('fails with 1 error, \'second:Expected 4 to be NaN.\'',
      function() {
    because('first').expect(4).not.toBeNaN();
    because('second').expect(4).toBeNaN();
  });

  it('fails with 1 error, \'second:Expected function to throw an exception.\'',
      function() {
    var emptyFunc = function() {};
    because('first').expect(
        function() { throw new Error('die'); }).toThrow();
    because('second').expect(emptyFunc).toThrow();
  });

  it('fails with 1 error, \'second:Expected false to be truthy\'',
      function() {
    because('first').expect(true).toBeTruthy();
    because('second').expect(false).toBeTruthy();
  });

  it('fails with 1 error, \'second:Expected 0 to equal 3.\'',
      function() {
    because('first').expect('first'.indexOf('s')).toEqual(3);
    because('second').expect('second'.indexOf('s')).toEqual(3);
  });

  it('fails with 1 error, \'second:Expected true to be falsy.\'',
      function() {
    because('first').expect(false).toBeFalsy();
    because('second').expect(true).toBeFalsy();
  });
});
