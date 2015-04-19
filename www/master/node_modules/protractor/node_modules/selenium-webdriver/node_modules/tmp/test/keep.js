var spawn = require('./spawn');

var keep = spawn.arg;

spawn.tmpFunction({ keep: keep }, function (err, name) {
  if (err) {
    spawn.err(err, spawn.exit);
  } else {
    spawn.out(name, spawn.exit);
  }
});
