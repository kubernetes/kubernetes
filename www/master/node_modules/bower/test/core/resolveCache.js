var path = require('path');
var mout = require('mout');
var rimraf = require('rimraf');
var fs = require('graceful-fs');
var Q = require('q');
var expect = require('expect.js');
var mkdirp = require('mkdirp');
var ResolveCache = require('../../lib/core/ResolveCache');
var defaultConfig = require('../../lib/config');
var cmd = require('../../lib/util/cmd');
var copy = require('../../lib/util/copy');
var md5 = require('../../lib/util/md5');

describe('ResolveCache', function () {
    var resolveCache;
    var testPackage = path.resolve(__dirname, '../assets/package-a');
    var tempPackage = path.resolve(__dirname, '../tmp/temp-package');
    var tempPackage2 = path.resolve(__dirname, '../tmp/temp2-package');
    var cacheDir = path.join(__dirname, '../tmp/temp-resolve-cache');

    before(function (next) {
        // Delete cache folder
        rimraf.sync(cacheDir);

        // Instantiate resolver cache
        resolveCache = new ResolveCache(defaultConfig({
            storage: {
                packages: cacheDir
            }
        }));

        // Checkout test package version 0.2.0
        cmd('git', ['checkout', '0.2.0'], { cwd: testPackage })
        .then(next.bind(next, null), next);
    });

    beforeEach(function () {
        // Reset in memory cache for each test
        resolveCache.reset();
    });

    after(function () {
        // Remove cache folder afterwards
        rimraf.sync(cacheDir);
    });

    describe('.constructor', function () {
        beforeEach(function () {
            // Delete temp folder
            rimraf.sync(tempPackage);
        });
        after(function () {
            // Delete temp folder
            rimraf.sync(tempPackage);
        });

        function initialize(cacheDir) {
            return new ResolveCache(defaultConfig({
                storage: {
                    packages: cacheDir
                }
            }));
        }

        it('should create the cache folder if it doesn\'t exists', function () {
            initialize(tempPackage);
            expect(fs.existsSync(tempPackage)).to.be(true);
        });

        it('should not error out if the cache folder already exists', function () {
            mkdirp.sync(tempPackage);
            initialize(tempPackage);
        });
    });

    describe('.store', function () {
        var oldFsRename = fs.rename;

        beforeEach(function (next) {
            // Restore oldFsRename
            fs.rename = oldFsRename;

            // Create a fresh copy of the test package into temp
            rimraf.sync(tempPackage);
            copy.copyDir(testPackage, tempPackage, { ignore: ['.git'] })
            .then(next.bind(next, null), next);
        });

        it('should move the canonical dir to source-md5/version/ folder if package meta has a version', function (next) {
            resolveCache.store(tempPackage, {
                name: 'foo',
                version: '1.0.0',
                _source: 'foo',
                _target: '*'
            })
            .then(function (dir) {
                expect(dir).to.equal(path.join(cacheDir, md5('foo'), '1.0.0'));
                expect(fs.existsSync(dir)).to.be(true);
                expect(fs.existsSync(path.join(dir, 'baz'))).to.be(true);
                expect(fs.existsSync(tempPackage)).to.be(false);

                next();
            })
            .done();
        });

        it('should move the canonical dir to source-md5/target/ folder if package meta has no version', function (next) {
            resolveCache.store(tempPackage, {
                name: 'foo',
                _source: 'foo',
                _target: 'some-branch'
            })
            .then(function (dir) {
                expect(dir).to.equal(path.join(cacheDir, md5('foo'), 'some-branch'));
                expect(fs.existsSync(dir)).to.be(true);
                expect(fs.existsSync(path.join(dir, 'baz'))).to.be(true);
                expect(fs.existsSync(tempPackage)).to.be(false);

                next();
            })
            .done();
        });

        it('should move the canonical dir to source-md5/_wildcard/ folder if package meta has no version and target is *', function (next) {
            resolveCache.store(tempPackage, {
                name: 'foo',
                _source: 'foo',
                _target: '*'
            })
            .then(function (dir) {
                expect(dir).to.equal(path.join(cacheDir, md5('foo'), '_wildcard'));
                expect(fs.existsSync(dir)).to.be(true);
                expect(fs.existsSync(path.join(dir, 'baz'))).to.be(true);
                expect(fs.existsSync(tempPackage)).to.be(false);

                next();
            })
            .done();
        });

        it('should read the package meta if not present', function (next) {
            var pkgMeta = path.join(tempPackage, '.bower.json');

            // Copy bower.json to .bower.json and add some props
            copy.copyFile(path.join(tempPackage, 'component.json'), pkgMeta)
            .then(function () {
                return Q.nfcall(fs.readFile, pkgMeta)
                .then(function (contents) {
                    var json = JSON.parse(contents.toString());

                    json._target = '~0.2.0';
                    json._source = 'git://github.com/bower/test-package.git';

                    return Q.nfcall(fs.writeFile, pkgMeta, JSON.stringify(json, null, '  '));
                });
            })
            // Store as usual
            .then(function () {
                return resolveCache.store(tempPackage);
            })
            .then(function (dir) {
                expect(dir).to.equal(path.join(cacheDir, md5('git://github.com/bower/test-package.git'), '0.2.0'));
                expect(fs.existsSync(dir)).to.be(true);
                expect(fs.existsSync(path.join(dir, 'baz'))).to.be(true);
                expect(fs.existsSync(tempPackage)).to.be(false);

                next();
            })
            .done();
        });

        it('should error out when reading the package meta if the file does not exist', function (next) {
            resolveCache.store(tempPackage)
            .then(function () {
                next(new Error('Should have failed'));
            }, function (err) {
                expect(err).to.be.an(Error);
                expect(err.code).to.equal('ENOENT');
                expect(err.message).to.contain(path.join(tempPackage, '.bower.json'));

                next();
            })
            .done();
        });

        it('should error out when reading an invalid package meta', function (next) {
            var pkgMeta = path.join(tempPackage, '.bower.json');

            return Q.nfcall(fs.writeFile, pkgMeta, 'w00t')
            .then(function () {
                return resolveCache.store(tempPackage)
                .then(function () {
                    next(new Error('Should have failed'));
                }, function (err) {
                    expect(err).to.be.an(Error);
                    expect(err.code).to.equal('EMALFORMED');
                    expect(err.message).to.contain(path.join(tempPackage, '.bower.json'));

                    next();
                });
            })
            .done();
        });

        it('should move the canonical dir, even if it is in a different drive', function (next) {
            var hittedMock = false;

            fs.rename = function (src, dest, cb) {
                hittedMock = true;

                setTimeout(function () {
                    var err = new Error();
                    err.code = 'EXDEV';
                    cb(err);
                }, 10);
            };

            resolveCache.store(tempPackage, {
                name: 'foo',
                _source: 'foobar',
                _target: 'some-branch'
            })
            .then(function (dir) {
                // Ensure mock was called
                expect(hittedMock).to.be(true);

                expect(dir).to.equal(path.join(cacheDir, md5('foobar'), 'some-branch'));
                expect(fs.existsSync(dir)).to.be(true);
                expect(fs.existsSync(path.join(dir, 'baz'))).to.be(true);
                expect(fs.existsSync(tempPackage)).to.be(false);

                next();
            })
            .done();
        });

        it('should update the in-memory cache', function (next) {
            // Feed the cache
            resolveCache.versions('test-in-memory')
            // Copy temp package to temp package  2
            .then(function () {
                return copy.copyDir(tempPackage, tempPackage2, { ignore: ['.git'] });
            })
            // Store the two packages
            .then(function () {
                return resolveCache.store(tempPackage, {
                    name: 'foo',
                    version: '1.0.0',
                    _source: 'test-in-memory',
                    _target: '*'
                });
            })
            .then(function () {
                return resolveCache.store(tempPackage2, {
                    name: 'foo',
                    version: '1.0.1',
                    _source: 'test-in-memory',
                    _target: '*'
                });
            })
            // Cache should have been updated
            .then(function () {
                return resolveCache.versions('test-in-memory')
                .then(function (versions) {
                    expect(versions).to.eql(['1.0.1', '1.0.0']);

                    next();
                });
            })
            .done();
        });

        it('should url encode target when storing to the fs', function (next) {
            resolveCache.store(tempPackage, {
                name: 'foo',
                _source: 'foo',
                _target: 'foo/bar'
            })
            .then(function (dir) {
                expect(dir).to.equal(path.join(cacheDir, md5('foo'), 'foo%2Fbar'));
                expect(fs.existsSync(dir)).to.be(true);
                expect(fs.existsSync(path.join(dir, 'baz'))).to.be(true);
                expect(fs.existsSync(tempPackage)).to.be(false);

                next();
            })
            .done();
        });

        it('should be possible to store two package at same time', function (next) {
            var store = resolveCache.store.bind(resolveCache, tempPackage, {
                name: 'foo',
                _source: 'foo',
                _target: 'foo/bar'
            });
            var store2 = resolveCache.store.bind(resolveCache, tempPackage2, {
                name: 'foo',
                _source: 'foo',
                _target: 'foo/bar'
            });

            Q.all([store(), store2()]).then(function (dirs) {
                var dir = dirs[0];
                expect(dir).to.equal(path.join(cacheDir, md5('foo'), 'foo%2Fbar'));
                expect(fs.existsSync(dir)).to.be(true);
                expect(fs.existsSync(path.join(dir, 'baz'))).to.be(true);
                expect(fs.existsSync(tempPackage)).to.be(false);
                expect(fs.existsSync(tempPackage2)).to.be(false);

                next();
            }).done();
        });
    });

    describe('.versions', function () {
        it('should resolve to an array', function (next) {
            resolveCache.versions(String(Math.random()))
            .then(function (versions) {
                expect(versions).to.be.an('array');
                next();
            })
            .done();
        });

        it('should ignore non-semver folders of the source', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.mkdirSync(path.join(sourceDir, '0.1.0'));
            fs.mkdirSync(path.join(sourceDir, 'foo'));

            resolveCache.versions(source)
            .then(function (versions) {
                expect(versions).to.not.contain('foo');
                expect(versions).to.contain('0.0.1');
                expect(versions).to.contain('0.1.0');
                next();
            })
            .done();
        });

        it('should order the versions', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.mkdirSync(path.join(sourceDir, '0.1.0'));
            fs.mkdirSync(path.join(sourceDir, '0.1.0-rc.1'));

            resolveCache.versions(source)
            .then(function (versions) {
                expect(versions).to.eql(['0.1.0', '0.1.0-rc.1', '0.0.1']);
                next();
            })
            .done();
        });

        it('should cache versions to speed-up subsequent calls', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));

            resolveCache.versions(source)
            .then(function () {
                // Remove folder
                rimraf.sync(sourceDir);

                return resolveCache.versions(source);
            })
            .then(function (versions) {
                expect(versions).to.eql(['0.0.1']);
                next();
            })
            .done();
        });
    });

    describe('.retrieve', function () {
        it('should resolve to empty if there are no packages for the requested source', function (next) {
            resolveCache.retrieve(String(Math.random()))
            .spread(function () {
                expect(arguments.length).to.equal(0);
                next();
            })
            .done();
        });

        it('should resolve to empty if there are no suitable packages for the requested target', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.mkdirSync(path.join(sourceDir, '0.1.0'));
            fs.mkdirSync(path.join(sourceDir, '0.1.9'));
            fs.mkdirSync(path.join(sourceDir, '0.2.0'));

            resolveCache.retrieve(source, '~0.3.0')
            .spread(function () {
                expect(arguments.length).to.equal(0);

                return resolveCache.retrieve(source, 'some-branch');
            })
            .spread(function () {
                expect(arguments.length).to.equal(0);

                next();
            })
            .done();
        });

        it('should remove invalid packages from the cache if their package meta is missing or invalid', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.mkdirSync(path.join(sourceDir, '0.1.0'));
            fs.mkdirSync(path.join(sourceDir, '0.1.9'));
            fs.mkdirSync(path.join(sourceDir, '0.2.0'));

            // Create an invalid package meta
            fs.writeFileSync(path.join(sourceDir, '0.2.0', '.bower.json'), 'w00t');

            resolveCache.retrieve(source, '~0.1.0')
            .spread(function () {
                var dirs = fs.readdirSync(sourceDir);

                expect(arguments.length).to.equal(0);
                expect(dirs).to.contain('0.0.1');
                expect(dirs).to.contain('0.2.0');
                next();
            })
            .done();
        });

        it('should resolve to the highest package that matches a range target, ignoring pre-releases', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);
            var json = { name: 'foo' };

            // Create some versions
            fs.mkdirSync(sourceDir);

            json.version = '0.0.1';
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.writeFileSync(path.join(sourceDir, '0.0.1', '.bower.json'), JSON.stringify(json, null, '  '));

            json.version = '0.1.0';
            fs.mkdirSync(path.join(sourceDir, '0.1.0'));
            fs.writeFileSync(path.join(sourceDir, '0.1.0', '.bower.json'), JSON.stringify(json, null, '  '));

            json.version = '0.1.0-rc.1';
            fs.mkdirSync(path.join(sourceDir, '0.1.0-rc.1'));
            fs.writeFileSync(path.join(sourceDir, '0.1.0-rc.1', '.bower.json'), JSON.stringify(json, null, '  '));

            json.version = '0.1.9';
            fs.mkdirSync(path.join(sourceDir, '0.1.9'));
            fs.writeFileSync(path.join(sourceDir, '0.1.9', '.bower.json'), JSON.stringify(json, null, '  '));

            json.version = '0.2.0';
            fs.mkdirSync(path.join(sourceDir, '0.2.0'));
            fs.writeFileSync(path.join(sourceDir, '0.2.0', '.bower.json'), JSON.stringify(json, null, '  '));

            resolveCache.retrieve(source, '~0.1.0')
            .spread(function (canonicalDir, pkgMeta) {
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.version).to.equal('0.1.9');
                expect(canonicalDir).to.equal(path.join(sourceDir, '0.1.9'));

                return resolveCache.retrieve(source, '*');
            })
            .spread(function (canonicalDir, pkgMeta) {
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.version).to.equal('0.2.0');
                expect(canonicalDir).to.equal(path.join(sourceDir, '0.2.0'));

                next();
            })
            .done();
        });

        it('should resolve to the highest package that matches a range target, not ignoring pre-releases if they are the only versions', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);
            var json = { name: 'foo' };

            // Create some versions
            fs.mkdirSync(sourceDir);

            json.version = '0.1.0-rc.1';
            fs.mkdirSync(path.join(sourceDir, '0.1.0-rc.1'));
            fs.writeFileSync(path.join(sourceDir, '0.1.0-rc.1', '.bower.json'), JSON.stringify(json, null, '  '));

            json.version = '0.1.0-rc.2';
            fs.mkdirSync(path.join(sourceDir, '0.1.0-rc.2'));
            fs.writeFileSync(path.join(sourceDir, '0.1.0-rc.2', '.bower.json'), JSON.stringify(json, null, '  '));

            resolveCache.retrieve(source, '~0.1.0')
            .spread(function (canonicalDir, pkgMeta) {
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.version).to.equal('0.1.0-rc.2');
                expect(canonicalDir).to.equal(path.join(sourceDir, '0.1.0-rc.2'));

                next();
            })
            .done();
        });

        it('should resolve to exact match (including build metadata) if available', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);
            var json = { name: 'foo' };
            var encoded;

            // Create some versions
            fs.mkdirSync(sourceDir);

            json.version = '0.1.0';
            fs.mkdirSync(path.join(sourceDir, '0.1.0'));
            fs.writeFileSync(path.join(sourceDir, '0.1.0', '.bower.json'), JSON.stringify(json, null, '  '));

            json.version = '0.1.0+build.4';
            encoded = encodeURIComponent('0.1.0+build.4');
            fs.mkdirSync(path.join(sourceDir, encoded));
            fs.writeFileSync(path.join(sourceDir, encoded, '.bower.json'), JSON.stringify(json, null, '  '));

            json.version = '0.1.0+build.5';
            encoded = encodeURIComponent('0.1.0+build.5');
            fs.mkdirSync(path.join(sourceDir, encoded));
            fs.writeFileSync(path.join(sourceDir, encoded, '.bower.json'), JSON.stringify(json, null, '  '));

            json.version = '0.1.0+build.6';
            encoded = encodeURIComponent('0.1.0+build.6');
            fs.mkdirSync(path.join(sourceDir, encoded));
            fs.writeFileSync(path.join(sourceDir, encoded, '.bower.json'), JSON.stringify(json, null, '  '));

            resolveCache.retrieve(source, '0.1.0+build.5')
            .spread(function (canonicalDir, pkgMeta) {
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.version).to.equal('0.1.0+build.5');
                expect(canonicalDir).to.equal(path.join(sourceDir, encodeURIComponent('0.1.0+build.5')));

                next();
            })
            .done();
        });

        it('should resolve to the _wildcard package if target is * and there are no semver versions', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);
            var json = { name: 'foo' };

            // Create some versions
            fs.mkdirSync(sourceDir);

            fs.mkdirSync(path.join(sourceDir, '_wildcard'));
            fs.writeFileSync(path.join(sourceDir, '_wildcard', '.bower.json'), JSON.stringify(json, null, '  '));

            resolveCache.retrieve(source, '*')
            .spread(function (canonicalDir, pkgMeta) {
                expect(pkgMeta).to.be.an('object');
                expect(canonicalDir).to.equal(path.join(sourceDir, '_wildcard'));

                next();
            })
            .done();
        });

        it('should resolve to the exact target it\'s not a semver range', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);
            var json = { name: 'foo' };

            // Create some versions
            fs.mkdirSync(sourceDir);

            fs.mkdirSync(path.join(sourceDir, 'some-branch'));
            fs.writeFileSync(path.join(sourceDir, 'some-branch', '.bower.json'), JSON.stringify(json, null, '  '));

            fs.mkdirSync(path.join(sourceDir, 'other-branch'));
            fs.writeFileSync(path.join(sourceDir, 'other-branch', '.bower.json'), JSON.stringify(json, null, '  '));

            resolveCache.retrieve(source, 'some-branch')
            .spread(function (canonicalDir, pkgMeta) {
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta).to.not.have.property('version');

                next();
            })
            .done();
        });
    });

    describe('.eliminate', function () {
        beforeEach(function () {
            mkdirp.sync(cacheDir);
        });

        it('should delete the source-md5/version folder', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.mkdirSync(path.join(sourceDir, '0.1.0'));

            resolveCache.eliminate({
                name: 'foo',
                version: '0.0.1',
                _source: source,
                _target: '*'
            })
            .then(function () {
                expect(fs.existsSync(path.join(sourceDir, '0.0.1'))).to.be(false);
                expect(fs.existsSync(path.join(sourceDir, '0.1.0'))).to.be(true);

                next();
            })
            .done();
        });

        it('should delete the source-md5/target folder', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.mkdirSync(path.join(sourceDir, 'some-branch'));

            resolveCache.eliminate({
                name: 'foo',
                _source: source,
                _target: 'some-branch'
            })
            .then(function () {
                expect(fs.existsSync(path.join(sourceDir, 'some-branch'))).to.be(false);
                expect(fs.existsSync(path.join(sourceDir, '0.0.1'))).to.be(true);

                next();
            })
            .done();
        });

        it('should delete the source-md5/_wildcard folder', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.mkdirSync(path.join(sourceDir, '_wildcard'));

            resolveCache.eliminate({
                name: 'foo',
                _source: source,
                _target: '*'
            })
            .then(function () {
                expect(fs.existsSync(path.join(sourceDir, '_wildcard'))).to.be(false);
                expect(fs.existsSync(path.join(sourceDir, '0.0.1'))).to.be(true);

                next();
            })
            .done();
        });

        it('should delete the source-md5 folder if empty', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));

            resolveCache.eliminate({
                name: 'foo',
                version: '0.0.1',
                _source: source,
                _target: '*'
            })
            .then(function () {
                expect(fs.existsSync(path.join(sourceDir, '0.0.1'))).to.be(false);
                expect(fs.existsSync(path.join(sourceDir))).to.be(false);

                next();
            })
            .done();
        });

        it('should remove entry from in memory cache if the source-md5 folder was deleted', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));

            // Feed up the cache
            resolveCache.versions(source)
            // Eliminate
            .then(function () {
                return resolveCache.eliminate({
                    name: 'foo',
                    version: '0.0.1',
                    _source: source,
                    _target: '*'
                });
            })
            .then(function () {
                // At this point the parent folder should be deleted
                // To test against the in-memory cache, we create a folder
                // manually and request the versions
                mkdirp.sync(path.join(sourceDir, '0.0.2'));

                resolveCache.versions(source)
                .then(function (versions) {
                    expect(versions).to.eql(['0.0.2']);

                    next();
                });
            })
            .done();
        });
    });

    describe('.clear', function () {
        beforeEach(function () {
            mkdirp.sync(cacheDir);
        });

        it('should empty the whole cache folder', function (next) {
            resolveCache.clear()
            .then(function () {
                var files;

                expect(fs.existsSync(cacheDir)).to.be(true);

                files = fs.readdirSync(cacheDir);
                expect(files.length).to.be(0);

                next();
            })
            .done();
        });

        it('should erase the in-memory cache', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));

            // Feed the in-memory cache
            resolveCache.versions(source)
            // Clear
            .then(function () {
                return resolveCache.clear();
            })
            .then(function () {
                // To test against the in-memory cache, we create a folder
                // manually and request the versions
                mkdirp.sync(path.join(sourceDir, '0.0.2'));

                resolveCache.versions(source)
                .then(function (versions) {
                    expect(versions).to.eql(['0.0.2']);

                    next();
                });
            })
            .done();
        });
    });

    describe('.reset', function () {
        it('should erase the in-memory cache', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));

            // Feed the in-memory cache
            resolveCache.versions(source)
            .then(function () {
                // Delete 0.0.1 and create 0.0.2
                fs.rmdirSync(path.join(sourceDir, '0.0.1'));
                fs.mkdirSync(path.join(sourceDir, '0.0.2'));

                // Reset cache
                resolveCache.reset();

                // Get versions
                return resolveCache.versions(source);
            })
            .then(function (versions) {
                expect(versions).to.eql(['0.0.2']);

                next();
            })
            .done();
        });
    });

    describe('.list', function () {
        beforeEach(function () {
            rimraf.sync(cacheDir);
            mkdirp.sync(cacheDir);
        });

        it('should resolve to an empty array if the cache is empty', function (next) {
            resolveCache.list()
            .then(function (entries) {
                expect(entries).to.be.an('array');
                expect(entries.length).to.be(0);

                next();
            })
            .done();
        });

        it('should resolve to an ordered array of entries (name ASC, release ASC)', function (next) {
            var source = 'list-package-1';
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            var source2 = 'list-package-2';
            var sourceId2 = md5(source2);
            var sourceDir2 = path.join(cacheDir, sourceId2);

            var json = {
                name: 'foo'
            };

            // Create some versions for different sources
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            json.version = '0.0.1';
            fs.writeFileSync(path.join(sourceDir, '0.0.1', '.bower.json'), JSON.stringify(json, null, '  '));

            fs.mkdirSync(path.join(sourceDir, '0.1.0'));
            json.version = '0.1.0';
            fs.writeFileSync(path.join(sourceDir, '0.1.0', '.bower.json'), JSON.stringify(json, null, '  '));

            delete json.version;

            fs.mkdirSync(path.join(sourceDir, 'foo'));
            json._target = 'foo';
            fs.writeFileSync(path.join(sourceDir, 'foo', '.bower.json'), JSON.stringify(json, null, '  '));

            fs.mkdirSync(path.join(sourceDir, 'bar'));
            json._target = 'bar';
            fs.writeFileSync(path.join(sourceDir, 'bar', '.bower.json'), JSON.stringify(json, null, '  '));

            fs.mkdirSync(path.join(sourceDir, 'aa'));
            json._target = 'aa';
            fs.writeFileSync(path.join(sourceDir, 'aa', '.bower.json'), JSON.stringify(json, null, '  '));

            delete json._target;

            fs.mkdirSync(sourceDir2);
            fs.mkdirSync(path.join(sourceDir2, '0.2.1'));
            json.version = '0.2.1';
            fs.writeFileSync(path.join(sourceDir2, '0.2.1', '.bower.json'), JSON.stringify(json, null, '  '));

            fs.mkdirSync(path.join(sourceDir2, '0.2.0'));
            json.name = 'abc';
            json.version = '0.2.0';
            fs.writeFileSync(path.join(sourceDir2, '0.2.0', '.bower.json'), JSON.stringify(json, null, '  '));

            resolveCache.list()
            .then(function (entries) {
                var expectedJson;
                var bowerDir = path.join(__dirname, '../..');

                expect(entries).to.be.an('array');

                expectedJson = fs.readFileSync(path.join(__dirname, '../assets/resolve-cache/list-json-1.json'));

                mout.object.forOwn(entries, function (entry) {
                    // Trim absolute bower path from json
                    entry.canonicalDir = entry.canonicalDir.substr(bowerDir.length);
                    // Convert windows \ paths to /
                    entry.canonicalDir = entry.canonicalDir.replace(/\\/g, '/');
                });

                expect(entries).to.eql(JSON.parse(expectedJson));

                next();
            })
            .done();
        });

        it('should ignore lurking files where dirs are expected', function (next) {
            var source = 'list-package-1';
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);
            var json = {
                name: 'foo'
            };

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            json.version = '0.0.1';
            fs.writeFileSync(path.join(sourceDir, '0.0.1', '.bower.json'), JSON.stringify(json, null, '  '));

            // Create lurking files
            fs.writeFileSync(path.join(cacheDir, 'foo'), 'w00t');
            fs.writeFileSync(path.join(cacheDir, '.DS_Store'), '');
            fs.writeFileSync(path.join(sourceDir, 'foo'), 'w00t');
            fs.writeFileSync(path.join(sourceDir, '.DS_Store'), '');

            // It should not error out
            resolveCache.list()
            .then(function (entries) {
                expect(entries).to.be.an('array');
                expect(entries.length).to.be(1);
                expect(entries[0].pkgMeta).to.eql(json);

                // Lurking file should have been removed
                expect(fs.existsSync(path.join(cacheDir, 'foo'))).to.be(false);
                expect(fs.existsSync(path.join(cacheDir, '.DS_Store'))).to.be(false);
                expect(fs.existsSync(path.join(sourceDir, 'foo'))).to.be(false);
                expect(fs.existsSync(path.join(sourceDir, '.DS_Store'))).to.be(false);

                next();
            })
            .done();

        });

        it('should delete entries if failed to read package meta', function (next) {
            var source = 'list-package-1';
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);
            var json = {
                name: 'foo'
            };

            // Create invalid versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));

            fs.mkdirSync(path.join(sourceDir, '0.0.2'));
            fs.writeFileSync(path.join(sourceDir, '0.0.2', '.bower.json'), 'w00t');

            // Create valid version
            fs.mkdirSync(path.join(sourceDir, '0.0.3'));
            json.version = '0.0.3';
            fs.writeFileSync(path.join(sourceDir, '0.0.3', '.bower.json'), JSON.stringify(json, null, '  '));

            // It should not error out
            resolveCache.list()
            .then(function (entries) {
                expect(entries).to.be.an('array');
                expect(entries.length).to.be(1);
                expect(entries[0].pkgMeta).to.eql(json);

                // Packages with invalid metas should have been removed
                expect(fs.existsSync(path.join(sourceDir, '0.0.1'))).to.be(false);
                expect(fs.existsSync(path.join(sourceDir, '0.0.2'))).to.be(false);

                next();
            })
            .done();
        });
    });

    describe('#clearRuntimeCache', function () {
        it('should clear the in-memory cache for all sources', function (next) {
            var source = String(Math.random());
            var sourceId = md5(source);
            var sourceDir = path.join(cacheDir, sourceId);

            var source2 = String(Math.random());
            var sourceId2 = md5(source2);
            var sourceDir2 = path.join(cacheDir, sourceId2);

            // Create some versions
            fs.mkdirSync(sourceDir);
            fs.mkdirSync(path.join(sourceDir, '0.0.1'));
            fs.mkdirSync(sourceDir2);
            fs.mkdirSync(path.join(sourceDir2, '0.0.2'));

            // Feed the cache
            resolveCache.versions(source)
            .then(function () {
                return resolveCache.versions(source2);
            })
            .then(function () {
                // Create some more
                fs.mkdirSync(path.join(sourceDir, '0.0.3'));
                fs.mkdirSync(path.join(sourceDir2, '0.0.4'));

                // Reset cache
                ResolveCache.clearRuntimeCache();
            })
            .then(function () {
                return resolveCache.versions(source)
                .then(function (versions) {
                    expect(versions).to.eql(['0.0.3', '0.0.1']);

                    return resolveCache.versions(source2);
                })
                .then(function (versions) {
                    expect(versions).to.eql(['0.0.4', '0.0.2']);

                    next();
                });
            })
            .done();
        });
    });
});
