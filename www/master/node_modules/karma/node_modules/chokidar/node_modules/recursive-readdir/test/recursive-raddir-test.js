var assert = require('assert')
var readdir = require('../index')

describe('readdir', function() {
  it('correctly lists all files in nested directories', function (done) {
    var expectedFiles = [__dirname + '/testdir/a/a', __dirname + '/testdir/a/beans',
      __dirname + '/testdir/b/123', __dirname + '/testdir/b/b/hurp-durp',
      __dirname + '/testdir/c.txt', __dirname + '/testdir/d.txt'
    ]

    readdir(__dirname + '/testdir', function(err, list) {
      assert.ifError(err);
      assert.deepEqual(list.sort(), expectedFiles.sort());
      done()
    })
  })
})
