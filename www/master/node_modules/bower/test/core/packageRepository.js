var expect = require('expect.js');
var Q = require('q');
var path = require('path');
var mout = require('mout');
var fs = require('graceful-fs');
var rimraf = require('rimraf');
var RegistryClient = require('bower-registry-client');
var Logger = require('bower-logger');
var proxyquire = require('proxyquire');
var defaultConfig = require('../../lib/config');
var ResolveCache = require('../../lib/core/ResolveCache');
var resolvers = require('../../lib/core/resolvers');
var copy = require('../../lib/util/copy');
var helpers = require('../helpers');

describe('PackageRepository', function () {
    var packageRepository;
    var resolver;
    var resolverFactoryHook;
    var resolverFactoryClearHook;
    var testPackage = path.resolve(__dirname, '../assets/package-a');
    var tempPackage = path.resolve(__dirname, '../tmp/temp-package');
    var packagesCacheDir = path.join(__dirname, '../tmp/temp-resolve-cache');
    var registryCacheDir = path.join(__dirname, '../tmp/temp-registry-cache');
    var mockSource = helpers.localSource(testPackage);

    var forceCaching = true;

    after(function () {
        rimraf.sync(registryCacheDir);
        rimraf.sync(packagesCacheDir);
    });

    beforeEach(function (next) {
        var PackageRepository;
        var config;
        var logger = new Logger();

        // Config
        config = defaultConfig({
            storage: {
                packages: packagesCacheDir,
                registry: registryCacheDir
            }
        });

        // Mock the resolver factory to always return a resolver for the test package
        function resolverFactory(decEndpoint, _config, _logger, _registryClient) {
            expect(_config).to.eql(config);
            expect(_logger).to.be.an(Logger);
            expect(_registryClient).to.be.an(RegistryClient);

            decEndpoint = mout.object.deepMixIn({}, decEndpoint);
            decEndpoint.source = mockSource;

            resolver = new resolvers.GitRemote(decEndpoint, _config, _logger);

            if (forceCaching) {
                // Force to use cache even for local resources
                resolver.isCacheable = function () {
                    return true;
                };
            }

            resolverFactoryHook(resolver);

            return Q.resolve(resolver);
        }
        resolverFactory.getConstructor = function () {
            return Q.resolve([resolvers.GitRemote, helpers.localSource(testPackage), false]);
        };
        resolverFactory.clearRuntimeCache = function () {
            resolverFactoryClearHook();
        };

        PackageRepository = proxyquire('../../lib/core/PackageRepository', {
            './resolverFactory': resolverFactory
        });
        packageRepository = new PackageRepository(config, logger);

        // Reset hooks
        resolverFactoryHook = resolverFactoryClearHook = function () {};

        // Remove temp package
        rimraf.sync(tempPackage);

        // Clear the repository
        packageRepository.clear()
        .then(next.bind(next, null), next);
    });

    describe('.constructor', function () {
        it('should pass the config correctly to the registry client, including its cache folder', function () {
            expect(packageRepository._registryClient._config.cache).to.equal(registryCacheDir);
        });
    });

    describe('.fetch', function () {
        it('should call the resolver factory to get the appropriate resolver', function (next) {
            var called;

            resolverFactoryHook = function () {
                called = true;
            };

            packageRepository.fetch({ name: '', source: 'foo', target: '~0.1.0' })
            .spread(function (canonicalDir, pkgMeta) {
                expect(called).to.be(true);
                expect(fs.existsSync(canonicalDir)).to.be(true);
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.name).to.be('package-a');
                expect(pkgMeta.version).to.be('0.1.1');
                next();
            })
            .done();
        });

        it('should just call the resolver resolve method if force was specified', function (next) {
            var called = [];

            resolverFactoryHook = function (resolver) {
                var originalResolve = resolver.resolve;

                resolver.resolve = function () {
                    called.push('resolve');
                    return originalResolve.apply(this, arguments);
                };

                resolver.hasNew = function () {
                    called.push('hasNew');
                    return Q.resolve(false);
                };
            };

            packageRepository._resolveCache.retrieve = function () {
                called.push('retrieve');
                return Q.resolve([]);
            };

            packageRepository._config.force = true;
            packageRepository.fetch({ name: '', source: 'foo', target: ' ~0.1.0' })
            .spread(function (canonicalDir, pkgMeta) {
                expect(called).to.eql(['resolve']);
                expect(fs.existsSync(canonicalDir)).to.be(true);
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.name).to.be('package-a');
                expect(pkgMeta.version).to.be('0.1.1');
                next();
            })
            .done();
        });

        it('should attempt to retrieve a resolved package from the resolve package', function (next) {
            var called = false;
            var originalRetrieve = packageRepository._resolveCache.retrieve;

            packageRepository._resolveCache.retrieve = function (source) {
                called = true;
                expect(source).to.be(mockSource);
                return originalRetrieve.apply(this, arguments);
            };

            packageRepository.fetch({ name: '', source: 'foo', target: '~0.1.0' })
            .spread(function (canonicalDir, pkgMeta) {
                expect(called).to.be(true);
                expect(fs.existsSync(canonicalDir)).to.be(true);
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.name).to.be('package-a');
                expect(pkgMeta.version).to.be('0.1.1');
                next();
            })
            .done();
        });

        it('should avoid using cache for local resources', function (next) {
            forceCaching = false;

            var called = false;
            var originalRetrieve = packageRepository._resolveCache.retrieve;

            packageRepository._resolveCache.retrieve = function (source) {
                called = true;
                expect(source).to.be(mockSource);
                return originalRetrieve.apply(this, arguments);
            };

            packageRepository.fetch({ name: '', source: helpers.localSource(testPackage), target: '~0.1.0' })
            .spread(function (canonicalDir, pkgMeta) {
                expect(called).to.be(false);
                expect(fs.existsSync(canonicalDir)).to.be(true);
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.name).to.be('package-a');
                expect(pkgMeta.version).to.be('0.1.1');
                forceCaching = true;
                next();
            })
            .done();
        });

        it('should just call the resolver resolve method if no appropriate package was found in the resolve cache', function (next) {
            var called = [];

            resolverFactoryHook = function (resolver) {
                var originalResolve = resolver.resolve;

                resolver.resolve = function () {
                    called.push('resolve');
                    return originalResolve.apply(this, arguments);
                };

                resolver.hasNew = function () {
                    called.push('hasNew');
                };
            };

            packageRepository._resolveCache.retrieve = function () {
                return Q.resolve([]);
            };

            packageRepository.fetch({ name: '', source: 'foo', target: ' ~0.1.0' })
            .spread(function (canonicalDir, pkgMeta) {
                expect(called).to.eql(['resolve']);
                expect(fs.existsSync(canonicalDir)).to.be(true);
                expect(pkgMeta).to.be.an('object');
                expect(pkgMeta.name).to.be('package-a');
                expect(pkgMeta.version).to.be('0.1.1');
                next();
            })
            .done();
        });

        it('should call the resolver hasNew method if an appropriate package was found in the resolve cache', function (next) {
            var json = {
                name: 'a',
                version: '0.2.1'
            };
            var called;

            resolverFactoryHook = function (resolver) {
                var originalHasNew = resolver.hasNew;

                resolver.hasNew = function (canonicalDir, pkgMeta) {
                    expect(canonicalDir).to.equal(tempPackage);
                    expect(pkgMeta).to.eql(json);
                    called = true;
                    return originalHasNew.apply(this, arguments);
                };
            };

            packageRepository._resolveCache.retrieve = function () {
                return Q.resolve([tempPackage, json]);
            };

            copy.copyDir(testPackage, tempPackage, { ignore: ['.git'] })
            .then(function () {
                fs.writeFileSync(path.join(tempPackage, '.bower.json'), JSON.stringify(json));

                return packageRepository.fetch({ name: '', source: 'foo', target: '~0.1.0' })
                .spread(function (canonicalDir, pkgMeta) {
                    expect(called).to.be(true);
                    expect(fs.existsSync(canonicalDir)).to.be(true);
                    expect(pkgMeta).to.be.an('object');
                    expect(pkgMeta.name).to.be('package-a');
                    expect(pkgMeta.version).to.be('0.1.1');
                    next();
                });
            })
            .done();
        });

        it('should call the resolver resolve method if hasNew resolved to true', function (next) {
            var json = {
                name: 'a',
                version: '0.2.0'
            };
            var called = [];

            resolverFactoryHook = function (resolver) {
                var originalResolve = resolver.resolve;

                resolver.resolve = function () {
                    called.push('resolve');
                    return originalResolve.apply(this, arguments);
                };

                resolver.hasNew = function (canonicalDir, pkgMeta) {
                    expect(canonicalDir).to.equal(tempPackage);
                    expect(pkgMeta).to.eql(json);
                    called.push('hasNew');
                    return Q.resolve(true);
                };
            };

            packageRepository._resolveCache.retrieve = function () {
                return Q.resolve([tempPackage, json]);
            };

            copy.copyDir(testPackage, tempPackage, { ignore: ['.git'] })
            .then(function () {
                fs.writeFileSync(path.join(tempPackage, '.bower.json'), JSON.stringify(json));

                return packageRepository.fetch({ name: '', source: 'foo', target: '~0.2.0' })
                .spread(function (canonicalDir, pkgMeta) {
                    expect(called).to.eql(['hasNew', 'resolve']);
                    expect(fs.existsSync(canonicalDir)).to.be(true);
                    expect(pkgMeta).to.be.an('object');
                    expect(pkgMeta.name).to.be('a');
                    expect(pkgMeta.version).to.be('0.2.2');
                    next();
                });
            })
            .done();
        });

        it('should resolve to the cached package if hasNew resolve to false', function (next) {
            var json = {
                name: 'a',
                version: '0.2.0'
            };
            var called = [];

            resolverFactoryHook = function (resolver) {
                var originalResolve = resolver.resolve;

                resolver.resolve = function () {
                    called.push('resolve');
                    return originalResolve.apply(this, arguments);
                };

                resolver.hasNew = function (canonicalDir, pkgMeta) {
                    expect(canonicalDir).to.equal(tempPackage);
                    expect(pkgMeta).to.eql(json);
                    called.push('hasNew');
                    return Q.resolve(false);
                };
            };

            packageRepository._resolveCache.retrieve = function () {
                return Q.resolve([tempPackage, json]);
            };

            copy.copyDir(testPackage, tempPackage, { ignore: ['.git'] })
            .then(function () {
                fs.writeFileSync(path.join(tempPackage, '.bower.json'), JSON.stringify(json));

                return packageRepository.fetch({ name: '', source: 'foo', target: '~0.2.0' })
                .spread(function (canonicalDir, pkgMeta) {
                    expect(called).to.eql(['hasNew']);
                    expect(canonicalDir).to.equal(tempPackage);
                    expect(pkgMeta).to.eql(json);
                    next();
                });
            })
            .done();
        });

        it('should just use the cached package if offline was specified', function (next) {
            var json = {
                name: 'a',
                version: '0.2.0'
            };
            var called = [];

            resolverFactoryHook = function (resolver) {
                var originalResolve = resolver.resolve;

                resolver.hasNew = function (canonicalDir, pkgMeta) {
                    expect(canonicalDir).to.equal(tempPackage);
                    expect(pkgMeta).to.eql(json);
                    called.push('resolve');
                    return originalResolve.apply(this, arguments);
                };

                resolver.hasNew = function () {
                    called.push('hasNew');
                    return Q.resolve(false);
                };
            };

            packageRepository._resolveCache.retrieve = function () {
                return Q.resolve([tempPackage, json]);
            };

            copy.copyDir(testPackage, tempPackage, { ignore: ['.git'] })
            .then(function () {
                fs.writeFileSync(path.join(tempPackage, '.bower.json'), JSON.stringify(json));

                packageRepository._config.offline = true;
                return packageRepository.fetch({ name: '', source: 'foo', target: '~0.2.0' })
                .spread(function (canonicalDir, pkgMeta) {
                    expect(called.length).to.be(0);
                    expect(canonicalDir).to.equal(tempPackage);
                    expect(pkgMeta).to.eql(json);
                    next();
                });
            })
            .done();
        });

        it('should error out if there is no appropriate package in the resolve cache and offline was specified', function (next) {
            packageRepository._config.offline = true;
            packageRepository.fetch({ name: '', source: 'foo', target: '~0.2.0' })
            .then(function () {
                throw new Error('Should have failed');
            }, function (err) {
                expect(err).to.be.an(Error);
                expect(err.code).to.equal('ENOCACHE');

                next();
            })
            .done();
        });
    });

    describe('.versions', function () {
        it('should call the versions method on the concrete resolver', function (next) {
            var called = [];
            var originalVersions = resolvers.GitRemote.versions;

            resolvers.GitRemote.versions = function (source) {
                expect(source).to.equal(mockSource);
                called.push('resolver');
                return Q.resolve([]);
            };

            packageRepository._resolveCache.versions = function () {
                called.push('resolve-cache');
                return Q.resolve([]);
            };

            packageRepository.versions('foo')
            .then(function (versions) {
                expect(called).to.eql(['resolver']);
                expect(versions).to.be.an('array');
                expect(versions.length).to.be(0);

                next();
            })
            .fin(function () {
                resolvers.GitRemote.versions = originalVersions;
            })
            .done();
        });

        it('should call the versions method on the resolve cache if offline was specified', function (next) {
            var called = [];
            var originalVersions = resolvers.GitRemote.versions;

            resolvers.GitRemote.versions = function () {
                called.push('resolver');
                return Q.resolve([]);
            };

            packageRepository._resolveCache.versions = function (source) {
                expect(source).to.equal(mockSource);
                called.push('resolve-cache');
                return Q.resolve([]);
            };

            packageRepository._config.offline = true;
            packageRepository.versions('foo')
            .then(function (versions) {
                expect(called).to.eql(['resolve-cache']);
                expect(versions).to.be.an('array');
                expect(versions.length).to.be(0);

                next();
            })
            .fin(function () {
                resolvers.GitRemote.versions = originalVersions;
            })
            .done();
        });
    });

    describe('.eliminate', function () {
        it('should call the eliminate method from the resolve cache', function (next) {
            var called;
            var json = {
                name: 'a',
                version: '0.2.0',
                _source: 'foo'
            };

            packageRepository._resolveCache.eliminate = function (pkgMeta) {
                expect(pkgMeta).to.eql(json);
                called = true;
                return Q.resolve();
            };

            packageRepository.eliminate(json)
            .then(function () {
                expect(called).to.be(true);
                next();
            })
            .done();
        });

        it('should call the clearCache method with the name from the registry client', function (next) {
            var called;
            var json = {
                name: 'a',
                version: '0.2.0',
                _source: 'foo'
            };

            packageRepository._registryClient.clearCache = function (name, callback) {
                expect(name).to.eql(json.name);
                called = true;
                callback();
            };

            packageRepository.eliminate(json)
            .then(function () {
                expect(called).to.be(true);
                next();
            })
            .done();
        });
    });

    describe('.list', function () {
        it('should proxy to the resolve cache list method', function (next) {
            var called;
            var originalList = packageRepository._resolveCache.list;

            packageRepository._resolveCache.list = function () {
                called = true;
                return originalList.apply(this, arguments);
            };

            packageRepository.list()
            .then(function (entries) {
                expect(called).to.be(true);
                expect(entries).to.be.an('array');
                next();
            })
            .done();
        });
    });

    describe('.clear', function () {
        it('should call the clear method from the resolve cache', function (next) {
            var called;

            packageRepository._resolveCache.clear = function () {
                called = true;
                return Q.resolve();
            };

            packageRepository.clear()
            .then(function () {
                expect(called).to.be(true);
                next();
            })
            .done();
        });

        it('should call the clearCache method without name from the registry client', function (next) {
            var called;

            packageRepository._registryClient.clearCache = function (callback) {
                called = true;
                callback();
            };

            packageRepository.clear()
            .then(function () {
                expect(called).to.be(true);
                next();
            })
            .done();
        });
    });

    describe('.reset', function () {
        it('should call the reset method from the resolve cache', function () {
            var called;

            packageRepository._resolveCache.reset = function () {
                called = true;
                return packageRepository._resolveCache;
            };

            packageRepository.reset();
            expect(called).to.be(true);
        });

        it('should call the resetCache method without name from the registry client', function () {
            var called;

            packageRepository._registryClient.resetCache = function () {
                called = true;
                return packageRepository._registryClient;
            };

            packageRepository.reset();
            expect(called).to.be(true);
        });
    });

    describe('.getRegistryClient', function () {
        it('should return the underlying registry client', function () {
            expect(packageRepository.getRegistryClient()).to.be.an(RegistryClient);
        });
    });

    describe('.getResolveCache', function () {
        it('should return the underlying resolve cache', function () {
            expect(packageRepository.getResolveCache()).to.be.an(ResolveCache);
        });
    });

    describe('#clearRuntimeCache', function () {
        it('should clear the resolve cache runtime cache', function () {
            var called;
            var originalClearRuntimeCache = ResolveCache.clearRuntimeCache;

            // No need to restore the original method since the constructor
            // gets re-assigned every time in beforeEach
            ResolveCache.clearRuntimeCache = function () {
                called = true;
                return originalClearRuntimeCache.apply(ResolveCache, arguments);
            };

            packageRepository.constructor.clearRuntimeCache();
            expect(called).to.be(true);
        });

        it('should clear the resolver factory runtime cache', function () {
            var called;

            resolverFactoryClearHook = function () {
                called = true;
            };

            packageRepository.constructor.clearRuntimeCache();
            expect(called).to.be(true);
        });

        it('should clear the registry runtime cache', function () {
            var called;
            var originalClearRuntimeCache = RegistryClient.clearRuntimeCache;

            // No need to restore the original method since the constructor
            // gets re-assigned every time in beforeEach
            RegistryClient.clearRuntimeCache = function () {
                called = true;
                return originalClearRuntimeCache.apply(RegistryClient, arguments);
            };

            packageRepository.constructor.clearRuntimeCache();
            expect(called).to.be(true);
        });
    });
});
