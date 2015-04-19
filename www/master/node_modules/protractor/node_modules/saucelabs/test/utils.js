describe('utils', function () {
  var utils = require('../lib/utils');

  describe('#extend', function () {
    var extend = utils.extend;

    it('extends an object with the attributes of another object', function () {
      var obj = { a: 1, b: 2 };
      extend(obj, { c: 3 });
      obj.should.deep.equal({ a: 1, b: 2, c: 3 });
    });
    it('extends an object with the attributes of multiple other objects', function () {
      var obj = { a: 1 };
      extend(obj, { b: 2 }, { c: 3 });
      obj.should.deep.equal({ a: 1, b: 2, c: 3 });
    });
  });

  describe('#replace', function () {
    var replace = utils.replace;

    it('replaces the placeholders in a string', function () {
      var str = 'http://:username::password@saucelabs.com/rest/v1/:username/tunnels/:id';
      str = replace(str, { username: 'foo', password: 'bar' });
      str.should.equal('http://foo:bar@saucelabs.com/rest/v1/foo/tunnels/:id');
    });
  });
});
