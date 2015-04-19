var common       = require('../common');
var test         = require('utest');
var assert       = common.assert;
var File = common.require('file');

var file;
var now = new Date;
test('IncomingForm', {
  before: function() {
    file = new File({
      size: 1024,
      path: '/tmp/cat.png',
      name: 'cat.png',
      type: 'image/png',
      lastModifiedDate: now,
      filename: 'cat.png',
      mime: 'image/png'
    })
  },

  '#toJSON()': function() {
    var obj = file.toJSON();
    var len = Object.keys(obj).length;
    assert.equal(1024, obj.size);
    assert.equal('/tmp/cat.png', obj.path);
    assert.equal('cat.png', obj.name);
    assert.equal('image/png', obj.type);
    assert.equal('image/png', obj.mime);
    assert.equal('cat.png', obj.filename);
    assert.equal(now, obj.mtime);
    assert.equal(len, 8);
  }
});