var
  tmp = require('../lib/tmp'),
  spawn = require('./spawn');

var graceful = spawn.arg;

if (graceful) {
  tmp.setGracefulCleanup();
}

spawn.tmpFunction(function (err, name) {
  spawn.out(name, function () {
    throw new Error("Thrown on purpose");
  });
});
