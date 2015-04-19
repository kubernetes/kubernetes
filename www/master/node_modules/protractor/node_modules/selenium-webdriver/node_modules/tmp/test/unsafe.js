var
  fs    = require('fs'),
  join  = require('path').join,
  spawn = require('./spawn');

var unsafe = spawn.arg;
spawn.tmpFunction({ unsafeCleanup: unsafe }, function (err, name) {
  if (err) {
    spawn.err(err, spawn.exit);
    return;
  }

  try {
    // file that should be removed
    var fd = fs.openSync(join(name, 'should-be-removed.file'), 'w');
    fs.closeSync(fd);

    // in tree source
    var symlinkSource = join(__dirname, 'symlinkme');
    // testing target
    var symlinkTarget = join(name, 'symlinkme-target');

    // symlink that should be removed but the contents should be preserved.
    fs.symlinkSync(symlinkSource, symlinkTarget, 'dir');

    spawn.out(name, spawn.exit);
  } catch (e) {
    spawn.err(e.toString(), spawn.exit);
  }
});
