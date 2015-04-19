var fs = require('../');
var rimraf = require('rimraf');
var mkdirp = require('mkdirp');
var test = require('tap').test;
var p = require('path').resolve(__dirname, 'files');

// Make sure to reserve the stderr fd
process.stderr.write('');

var num = 4097;
var paths = new Array(num);

test('make files', function (t) {
  rimraf.sync(p);
  mkdirp.sync(p);

  for (var i = 0; i < num; ++i) {
    paths[i] = 'files/file-' + i;
    fs.writeFileSync(paths[i], 'content');
  }

  t.end();
})

test('read files', function (t) {
  // now read them
  var done = 0;
  for (var i = 0; i < num; ++i) {
    fs.readFile(paths[i], function(err, data) {
      if (err)
        throw err;

      ++done;
      if (done === num) {
        t.pass('success');
        t.end()
      }
    });
  }
});

test('cleanup', function (t) {
  rimraf.sync(p);
  t.end();
});
