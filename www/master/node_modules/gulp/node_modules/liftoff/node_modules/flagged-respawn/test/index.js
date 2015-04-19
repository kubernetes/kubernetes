const expect = require('chai').expect;
const exec = require('child_process').exec;

const reorder = require('../lib/reorder');
const flaggedRespawn = require('../');

describe('flaggedRespawn', function () {
  var flags = ['--harmony', '--use_strict']

  describe('reorder', function () {

    it('should re-order args, placing special flags first', function () {
      var needsRespawn = ['node', 'file.js', '--flag', '--harmony', 'command'];
      var noRespawnNeeded = ['node', 'bin/flagged-respawn', 'thing'];
      expect(reorder(flags, needsRespawn))
        .to.deep.equal(['node', '--harmony', 'file.js', '--flag', 'command']);
      expect(reorder(flags, noRespawnNeeded))
        .to.deep.equal(noRespawnNeeded);
    });

    it('should ignore special flags when they are in the correct position', function () {
      var args = ['node', '--harmony', 'file.js', '--flag'];
      expect(reorder(flags, reorder(flags, args))).to.deep.equal(args);
    });

  });

  describe('execute', function () {

    it('should throw if no flags are specified', function () {
      expect(function () { flaggedRespawn.execute(); }).to.throw;
    });

    it('should throw if no argv is specified', function () {
      expect(function () { flaggedRespawn.execute(flags); }).to.throw;
    });

    it('should respawn and pipe stderr/stdout to parent', function (done) {
      exec('node ./test/bin/respawner.js --harmony', function (err, stdout, stderr) {
        expect(stdout.replace(/[0-9]/g, '')).to.equal('Special flags found, respawning.\nRespawned to PID: \nRunning!\n');
        done();
      });
    });

    it('should respawn and pass exit code from child to parent', function (done) {
      exec('node ./test/bin/exit_code.js --harmony', function (err, stdout, stderr) {
        expect(err.code).to.equal(100);
        done();
      });
    });

    it.skip('should respawn; if child is killed, parent should exit with same signal', function (done) {
      // TODO: figure out why travis hates this
      exec('node ./test/bin/signal.js --harmony', function (err, stdout, stderr) {
        console.log('err', err);
        console.log('stdout', stdout);
        console.log('stderr', stderr);
        expect(err.signal).to.equal('SIGHUP');
        done();
      });
    });

    it('should call back with ready as true when respawn is not needed', function () {
      var argv = ['node', './test/bin/respawner'];
      flaggedRespawn(flags, argv, function (ready) {
        expect(ready).to.be.true;
      });
    });

    it('should call back with ready as false when respawn is needed', function () {
      var argv = ['node', './test/bin/respawner', '--harmony'];
      flaggedRespawn(flags, argv, function (ready) {
        expect(ready).to.be.false;
      });
    });

    it('should call back with the child process when ready', function () {
      var argv = ['node', './test/bin/respawner', '--harmony'];
      flaggedRespawn(flags, argv, function (ready, child) {
        expect(child.pid).to.not.equal(process.pid);
      });
    });

    it('should call back with own process when respawn not needed', function () {
      var argv = ['node', './test/bin/respawner'];
      flaggedRespawn(flags, argv, function (ready, child) {
        expect(child.pid).to.equal(process.pid);
      });
    });

  });

});
