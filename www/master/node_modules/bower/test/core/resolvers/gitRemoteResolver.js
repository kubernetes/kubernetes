var expect = require('expect.js');
var path = require('path');
var fs = require('graceful-fs');
var Logger = require('bower-logger');
var helpers = require('../../helpers');
var Q = require('q');
var mout = require('mout');
var multiline = require('multiline').stripIndent;
var GitRemoteResolver = require('../../../lib/core/resolvers/GitRemoteResolver');
var defaultConfig = require('../../../lib/config');

describe('GitRemoteResolver', function () {
    var testPackage = path.resolve(__dirname, '../../assets/package-a');
    var logger;

    before(function () {
        logger = new Logger();
    });

    afterEach(function () {
        logger.removeAllListeners();
    });

    function clearResolverRuntimeCache() {
        GitRemoteResolver.clearRuntimeCache();
    }

    function create(decEndpoint) {
        if (typeof decEndpoint === 'string') {
            decEndpoint = { source: decEndpoint };
        }

        return new GitRemoteResolver(decEndpoint, defaultConfig(), logger);
    }

    describe('.constructor', function () {
        it('should guess the name from the path', function () {
            var resolver;

            resolver = create('file://' + testPackage);
            expect(resolver.getName()).to.equal('package-a');

            resolver = create('git://github.com/twitter/bower.git');
            expect(resolver.getName()).to.equal('bower');

            resolver = create('git://github.com/twitter/bower');
            expect(resolver.getName()).to.equal('bower');

            resolver = create('git://github.com');
            expect(resolver.getName()).to.equal('github.com');
        });
    });

    describe('.resolve', function () {
        it('should checkout correctly if resolution is a branch', function (next) {
            var resolver = create({ source: 'file://' + testPackage, target: 'some-branch' });

            resolver.resolve()
            .then(function (dir) {
                expect(dir).to.be.a('string');

                var files = fs.readdirSync(dir);
                var fooContents;

                expect(files).to.contain('foo');
                expect(files).to.contain('baz');
                expect(files).to.contain('baz');

                fooContents = fs.readFileSync(path.join(dir, 'foo')).toString();
                expect(fooContents).to.equal('foo foo');

                next();
            })
            .done();
        });

        it('should checkout correctly if resolution is a tag', function (next) {
            var resolver = create({ source: 'file://' + testPackage, target: '~0.0.1' });

            resolver.resolve()
            .then(function (dir) {
                expect(dir).to.be.a('string');

                var files = fs.readdirSync(dir);

                expect(files).to.contain('foo');
                expect(files).to.contain('bar');
                expect(files).to.not.contain('baz');

                next();
            })
            .done();
        });

        it('should checkout correctly if resolution is a commit', function (next) {
            var resolver = create({ source: 'file://' + testPackage, target: 'bdf51ece75e20cf404e49286727b7e92d33e9ad0' });

            resolver.resolve()
            .then(function (dir) {
                expect(dir).to.be.a('string');

                var files = fs.readdirSync(dir);

                expect(files).to.not.contain('foo');
                expect(files).to.not.contain('bar');
                expect(files).to.not.contain('baz');
                expect(files).to.contain('.master');
                next();
            })
            .done();
        });

        describe('shallow cloning', function () {
            var gitRemoteResolverFactory;

            beforeEach(function () {
                gitRemoteResolverFactory = function (handler) {
                    return helpers.require('lib/core/resolvers/GitRemoteResolver', {
                        '../../util/cmd': handler
                    });
                };
            });

            it('should add --depth=1 when shallow cloning is supported', function (next) {
                var testSource = 'http://foo/bar.git';

                var MyGitRemoteResolver = gitRemoteResolverFactory(function (cmd, args) {
                    // The first git call fetches the tags for the provided source
                    if (mout.array.equals(args, ['ls-remote', '--tags', '--heads', testSource])) {
                        // Return list of commits, including one tag.
                        // The tag will be used for the clone call.
                        return Q.all([multiline(function () {/*
                         e4655d250f2a3f64ef2d712f25dafa60652bb93e refs/heads/some-branch
                         0a7daf646d4fd743b6ef701d63bdbe20eee422de refs/tags/0.0.1
                         */
                        })]);
                    }
                    else if (args[0] === 'clone') {
                        // Verify parameters of the clone call.
                        // In this case, the arguments need to contain "--depth 1".
                        expect(args).to.eql(['clone', 'http://foo/bar.git', '-b', '0.0.1', '--progress', '.', '--depth', 1]);

                        // In this case, only the stderr content is evaluated. Everything's fine as long as it
                        // does not contain any error description.
                        return Q.all(['stdout', 'stderr']);
                    }
                });

                // Mock the call, return true for this test.
                MyGitRemoteResolver.prototype._supportsShallowCloning = function () {
                    return Q.resolve(true);
                };

                var resolver = new MyGitRemoteResolver({ source: testSource }, defaultConfig(), logger);

                resolver.resolve().then(function () {
                    next();
                });
            });

            it('should not add --depth=1 when shallow cloning is not supported', function (next) {
                var testSource = 'http://foo/bar.git';

                var MyGitRemoteResolver = gitRemoteResolverFactory(function (cmd, args) {
                    // The first git call fetches the tags for the provided source
                    if (mout.array.equals(args, ['ls-remote', '--tags', '--heads', testSource])) {
                        // Return list of commits, including one tag.
                        // The tag will be used for the clone call.
                        return Q.all([multiline(function () {/*
                         e4655d250f2a3f64ef2d712f25dafa60652bb93e refs/heads/some-branch
                         0a7daf646d4fd743b6ef701d63bdbe20eee422de refs/tags/0.0.1
                         */
                        })]);
                    }
                    else if (args[0] === 'clone') {
                        // Verify parameters of the clone call.
                        // In this case, the arguments should not contain "--depth 1".
                        expect(args).to.eql(['clone', 'http://foo/bar.git', '-b', '0.0.1', '--progress', '.']);

                        // In this case, only the stderr content is evaluated. Everything's fine as long as it
                        // does not contain any error description.
                        return Q.all(['stdout', 'stderr']);
                    }
                });

                // Mock the call, return false for this test.
                MyGitRemoteResolver.prototype._supportsShallowCloning = function () {
                    return Q.resolve(false);
                };

                var resolver = new MyGitRemoteResolver({ source: testSource }, defaultConfig(), logger);

                resolver.resolve().then(function () {
                    next();
                });
            });
        });

        it.skip('should handle gracefully servers that do not support --depth=1');
        it.skip('should report progress when it takes too long to clone');
    });

    describe('#refs', function () {
        afterEach(clearResolverRuntimeCache);

        it('should resolve to the references of the remote repository', function (next) {
            GitRemoteResolver.refs('file://' + testPackage)
            .then(function (refs) {
                // Remove master and test only for the first 7 refs
                refs = refs.slice(1, 8);

                expect(refs).to.eql([
                    'e4655d250f2a3f64ef2d712f25dafa60652bb93e refs/heads/some-branch',
                    '0a7daf646d4fd743b6ef701d63bdbe20eee422de refs/tags/0.0.1',
                    '0791865e6f4b88f69fc35167a09a6f0626627765 refs/tags/0.0.2',
                    '2af02ac6ddeaac1c2f4bead8d6287ce54269c039 refs/tags/0.1.0',
                    '6ab264f1ba5bafa80fb0198183493e4d5b20804a refs/tags/0.1.1',
                    'c91ed7facbb695510e3e1ab86bac8b5ac159f4f3 refs/tags/0.2.0',
                    '8556e55c65722a351ca5fdce4f1ebe83ec3f2365 refs/tags/0.2.1'
                ]);
                next();
            })
            .done();
        });

        it('should cache the results', function (next) {
            var source = 'file://' + testPackage;

            GitRemoteResolver.refs(source)
            .then(function () {
                // Manipulate the cache and check if it resolves for the cached ones
                GitRemoteResolver._cache.refs.get(source).splice(0, 1);

                // Check if it resolver to the same array
                return GitRemoteResolver.refs('file://' + testPackage);
            })
            .then(function (refs) {
                // Test only for the first 7 refs
                refs = refs.slice(0, 7);

                expect(refs).to.eql([
                    'e4655d250f2a3f64ef2d712f25dafa60652bb93e refs/heads/some-branch',
                    '0a7daf646d4fd743b6ef701d63bdbe20eee422de refs/tags/0.0.1',
                    '0791865e6f4b88f69fc35167a09a6f0626627765 refs/tags/0.0.2',
                    '2af02ac6ddeaac1c2f4bead8d6287ce54269c039 refs/tags/0.1.0',
                    '6ab264f1ba5bafa80fb0198183493e4d5b20804a refs/tags/0.1.1',
                    'c91ed7facbb695510e3e1ab86bac8b5ac159f4f3 refs/tags/0.2.0',
                    '8556e55c65722a351ca5fdce4f1ebe83ec3f2365 refs/tags/0.2.1'
                ]);
                next();
            })
            .done();
        });
    });

    describe('#_supportsShallowCloning', function () {
        var gitRemoteResolverFactory;

        beforeEach(function () {
            gitRemoteResolverFactory = function (handler) {
                return helpers.require('lib/core/resolvers/GitRemoteResolver', {
                    '../../util/cmd': handler
                });
            };
        });

        function createCmdHandlerFn (testSource, stderr) {
            return function (cmd, args, options) {
                expect(cmd).to.be('git');
                expect(args).to.eql([ 'ls-remote', '--heads', testSource ]);
                expect(options.env.GIT_CURL_VERBOSE).to.be(2);

                return Q.all(['stdout', stderr]);
            };
        }

        it('should call ls-remote when using http protocol', function (next) {
            var testSource = 'http://foo/bar.git';

            var MyGitRemoteResolver = gitRemoteResolverFactory(
                createCmdHandlerFn(testSource, multiline(function () {/*
                    foo: bar
                    Content-Type: none
                    1234: 5678
                */}))
            );

            var resolver = new MyGitRemoteResolver({ source: testSource }, defaultConfig(), logger);

            resolver._shallowClone().then(function (shallowCloningSupported) {
                expect(shallowCloningSupported).to.be(false);

                next();
            });
        });

        it('should call ls-remote when using https protocol', function (next) {
            var testSource = 'https://foo/bar.git';

            var MyGitRemoteResolver = gitRemoteResolverFactory(
                createCmdHandlerFn(testSource, multiline(function () {/*
                    foo: bar
                    Content-Type: none
                    1234: 5678
                */}))
            );

            var resolver = new MyGitRemoteResolver({ source: testSource }, defaultConfig(), logger);

            resolver._shallowClone().then(function (shallowCloningSupported) {
                expect(shallowCloningSupported).to.be(false);

                next();
            });
        });

        it('should evaluate to false when the URL can not be parsed', function (next) {
            var testSource = 'grmblfjx///:::.git';

            var MyGitRemoteResolver = gitRemoteResolverFactory(
                createCmdHandlerFn(testSource, multiline(function () {/*
                    foo: bar
                    Content-Type: none
                    1234: 5678
                */}))
            );

            var resolver = new MyGitRemoteResolver({ source: testSource }, defaultConfig(), logger);

            resolver._shallowClone().then(function (shallowCloningSupported) {
                expect(shallowCloningSupported).to.be(false);

                next();
            }, function (err) {
                next(err);
            });
        });

        it('should evaluate to true when the smart content type is returned', function (next) {
            var testSource = 'https://foo/bar.git';

            var MyGitRemoteResolver = gitRemoteResolverFactory(
                createCmdHandlerFn(testSource, multiline(function () {/*
                    foo: bar
                    Content-Type: application/x-git-upload-pack-advertisement
                    1234: 5678
                */}))
            );

            var resolver = new MyGitRemoteResolver({ source: testSource }, defaultConfig(), logger);

            resolver._shallowClone().then(function (shallowCloningSupported) {
                expect(shallowCloningSupported).to.be(true);

                next();
            });
        });

        it('should cache hosts that support shallow cloning', function (next) {
            var testSource = 'https://foo/bar.git';

            var counter = 0;

            var MyGitRemoteResolver = gitRemoteResolverFactory(
                function (cmd, args, options) {
                    counter++;

                    if (counter === 1) {
                        expect(cmd).to.be('git');
                        expect(args).to.eql([ 'ls-remote', '--heads', testSource ]);
                        expect(options.env.GIT_CURL_VERBOSE).to.be(2);

                        return Q.all(['stdout', multiline(function () {/*
                         foo: bar
                         Content-Type: application/x-git-upload-pack-advertisement
                         1234: 5678
                         */
                        })]);
                    }
                    else {
                        return Q.reject(new Error('More calls than expected'));
                    }
                }
            );

            var resolver = new MyGitRemoteResolver({ source: testSource }, defaultConfig(), logger);

            resolver._shallowClone().then(function (shallowCloningSupported) {
                expect(shallowCloningSupported).to.be(true);

                var resolver2 = new MyGitRemoteResolver({ source: testSource }, defaultConfig(), logger);

                resolver2._shallowClone().then(function (shallowCloningSupported) {
                    expect(shallowCloningSupported).to.be(true);

                    next();
                }, function(err) {
                    next(err);
                });
            });
        });

        it('should cache hosts that support shallow cloning across multiple repos', function (next) {
            var testSource1 = 'https://foo/bar.git';
            var testSource2 = 'https://foo/barbaz.git';

            var counter = 0;

            var MyGitRemoteResolver = gitRemoteResolverFactory(
                function (cmd, args, options) {
                    counter++;

                    if (counter === 1) {
                        expect(cmd).to.be('git');
                        expect(args).to.eql([ 'ls-remote', '--heads', testSource1 ]);
                        expect(options.env.GIT_CURL_VERBOSE).to.be(2);

                        return Q.all(['stdout', multiline(function () {/*
                         foo: bar
                         Content-Type: application/x-git-upload-pack-advertisement
                         1234: 5678
                         */
                        })]);
                    }
                    else {
                        return Q.reject(new Error('More calls than expected'));
                    }
                }
            );

            var resolver = new MyGitRemoteResolver({ source: testSource1 }, defaultConfig(), logger);

            resolver._shallowClone().then(function (shallowCloningSupported) {
                expect(shallowCloningSupported).to.be(true);

                var resolver2 = new MyGitRemoteResolver({ source: testSource2 }, defaultConfig(), logger);

                resolver2._shallowClone().then(function (shallowCloningSupported) {
                    expect(shallowCloningSupported).to.be(true);

                    next();
                }, function(err) {
                    next(err);
                });
            });
        });

        it('should run separate checks for separate hosts ', function (next) {
            var testSource1 = 'https://foo/bar.git';
            var testSource2 = 'https://foo.bar.baz/barbaz.git';

            var counter = 0;

            var MyGitRemoteResolver = gitRemoteResolverFactory(
                function (cmd, args, options) {
                    counter++;

                    if (counter === 1) {
                        expect(cmd).to.be('git');
                        expect(args).to.eql([ 'ls-remote', '--heads', testSource1 ]);
                        expect(options.env.GIT_CURL_VERBOSE).to.be(2);

                        return Q.all(['stdout', multiline(function () {/*
                         foo: bar
                         Content-Type: application/x-git-upload-pack-advertisement
                         1234: 5678
                         */
                        })]);
                    }
                    else {
                        expect(cmd).to.be('git');
                        expect(args).to.eql([ 'ls-remote', '--heads', testSource2 ]);
                        expect(options.env.GIT_CURL_VERBOSE).to.be(2);

                        return Q.all(['stdout', multiline(function () {/*
                         foo: barbaz
                         Content-Type: application/x-git-upload-pack-advertisement
                         1234: 5678
                         */
                        })]);
                    }
                }
            );

            var resolver = new MyGitRemoteResolver({ source: testSource1 }, defaultConfig(), logger);

            resolver._shallowClone().then(function (shallowCloningSupported) {
                expect(shallowCloningSupported).to.be(true);

                var resolver2 = new MyGitRemoteResolver({ source: testSource2 }, defaultConfig(), logger);

                resolver2._shallowClone().then(function (shallowCloningSupported) {
                    expect(shallowCloningSupported).to.be(true);

                    next();
                }, function(err) {
                    next(err);
                });
            });
        });
    });
});
