'use strict';
var path = require('path');
var exec = require('child_process').exec;
var glob = require('glob');
var assert = require('chai').assert;
var tmp = require('tmp');
var assetsPath = path.join(__dirname, 'assets');
var DecompressZip = require('../lib/decompress-zip');

var samples = glob.sync('*/archive.zip', {cwd: assetsPath});

if (samples.length === 0) {
    console.log('No sample ZIP files were found. Run "grunt test-files" to download them.');
    process.exit(1);
}

describe('Smoke test', function () {
    it('should find the public interface', function () {
        assert.isFunction(DecompressZip, 'constructor is a function');
        assert.isFunction(DecompressZip.prototype.list, 'decompress.list is a function');
        assert.isFunction(DecompressZip.prototype.extract, 'decompress.extract is a function');
    });
});

describe('Extract', function () {
    describe('errors', function () {
        var tmpDir;

        before(function (done) {
            tmp.dir({unsafeCleanup: true}, function (err, dir) {
                if (err) {
                    throw err;
                }

                tmpDir = dir;
                done();
            });
        });

        it('should emit an error when the file does not exist', function (done) {
            var zip = new DecompressZip('/my/non/existant/file.zip');

            zip.on('extract', function () {
                assert(false, '"extract" event should not fire');
                done();
            });

            zip.on('error', function (error) {
                assert(true, '"error" event should fire');
                done();
            });

            zip.extract({path: tmpDir});
        });

        it('should emit an error when stripping deeper than the path structure', function (done) {
            var zip = new DecompressZip(path.join(assetsPath, samples[0]));

            zip.on('extract', function () {
                assert(false, '"extract" event should not fire');
                done();
            });

            zip.on('error', function (error) {
                assert(true, '"error" event should fire');
                done();
            });

            zip.extract({path: tmpDir, strip: 3});
        });

        it('should emit a progress event on each file', function (done) {
            var zip = new DecompressZip(path.join(assetsPath, samples[0]));
            var numProgressEvents = 0;
            var numTotalFiles = 921;

            zip.on('progress', function (i, numFiles) {
                assert.equal(numFiles, numTotalFiles, '"progress" event should include the correct number of files');
                assert(typeof i === 'number', '"progress" event should include the number of the current file');
                numProgressEvents++;
            });

            zip.on('extract', function () {
                assert(true, '"extract" event should fire');
                assert.equal(numProgressEvents, numTotalFiles, 'there should be a "progress" event for every file');
                done();
            });

            zip.on('error', function (error) {
                assert(false, '"error" event should not fire');
                done();
            });

            zip.extract({path: tmpDir});
        });
    });

    describe('directory creation', function () {
        var tmpDir;
        var rmdirSync;
        before(function (done) {
            tmp.dir({unsafeCleanup: true}, function (err, dir, cleanupCallback) {
                if (err) {
                    throw err;
                }

                tmpDir = dir;
                rmdirSync = cleanupCallback;
                done();
            });
        });

        it('should create necessary directories, even on 2nd run', function (done) {
            var zip = new DecompressZip(path.join(assetsPath, samples[0]));
            zip.on('error', done);
            zip.on('extract', function () {
                rmdirSync(tmpDir);
                var zip2 = new DecompressZip(path.join(assetsPath, samples[0]));
                zip2.on('error', done);
                zip2.on('extract', function () {
                    done();
                });
                zip2.extract({path: tmpDir});
            });

            zip.extract({path: tmpDir});
        });
    });

    samples.forEach(function (sample) {
        var extracted = path.join(path.dirname(sample), 'extracted');

        describe(sample, function () {
            var tmpDir;

            before(function (done) {
                tmp.dir({unsafeCleanup: true}, function (err, dir) {
                    if (err) {
                        throw err;
                    }

                    tmpDir = dir;
                    done();
                });
            });


            it('should extract without any errors', function (done) {
                this.timeout(60000);
                var zip = new DecompressZip(path.join(assetsPath, sample));

                zip.on('extract', function () {
                    assert(true, 'success callback should be called');
                    done();
                });

                zip.on('error', function () {
                    assert(false, 'error callback should not be called');
                    done();
                });

                zip.extract({path: tmpDir});
            });

            it('should have the same output files as expected', function (done) {
                exec('diff -qr ' + extracted + ' ' + tmpDir, {cwd: assetsPath}, function (err, stdout, stderr) {
                    if (err) {
                        if (err.code === 1) {
                            assert(false, 'output should match');
                        } else {
                            throw err;
                        }
                    }
                    assert.equal(stdout, '');
                    assert.equal(stderr, '');
                    done();
                });
            });
        });
    });
});
