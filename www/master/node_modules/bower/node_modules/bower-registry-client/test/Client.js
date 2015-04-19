var RegistryClient = require('../Client');
var fs = require('fs');
var expect = require('expect.js');
var md5 = require('../lib/util/md5');
var nock = require('nock');
var http = require('http');

describe('RegistryClient', function () {
    beforeEach(function () {
        this.uri = 'https://bower.herokuapp.com';
        this.timeoutVal = 5000;
        this.registry = new RegistryClient({
            strictSsl: false,
            timeout: this.timeoutVal
        });
        this.conf = {
            search: [this.uri],
            register: this.uri,
            publish: this.uri
        };
    });

    describe('Constructor', function () {
        describe('instantiating a client', function () {
            it('should provide an instance of RegistryClient', function () {
                expect(this.registry instanceof RegistryClient).to.be.ok;
            });

            it('should set default registry config', function () {
                expect(this.registry._config.registry).to.eql(this.conf);
            });

            it('should set default search config', function () {
                expect(this.registry._config.registry.search[0]).to.eql(this.uri);
            });

            it('should set default register config', function () {
                expect(this.registry._config.registry.register).to.eql(this.uri);
            });

            it('should set default publish config', function () {
                expect(this.registry._config.registry.publish).to.eql(this.uri);
            });

            it('should set default cache path config', function () {
                expect(typeof this.registry._config.cache === 'string').to.be.ok;
            });

            it('should set default timeout config', function () {
                expect(this.registry._config.timeout).to.eql(this.timeoutVal);
            });

            it('should set default strictSsl config', function () {
                expect(this.registry._config.strictSsl).to.be(false);
            });
        });

        it('should have a lookup prototype method', function () {
            expect(RegistryClient.prototype).to.have.property('lookup');
        });

        it('should have a search prototype method', function () {
            expect(RegistryClient.prototype).to.have.property('search');
        });

        it('should have a list prototype method', function () {
            expect(RegistryClient.prototype).to.have.property('list');
        });

        it('should have a register prototype method', function () {
            expect(RegistryClient.prototype).to.have.property('register');
        });

        it('should have a clearCache prototype method', function () {
            expect(RegistryClient.prototype).to.have.property('clearCache');
        });

        it('should have a resetCache prototype method', function () {
            expect(RegistryClient.prototype).to.have.property('resetCache');
        });

        it('should have a clearRuntimeCache static method', function () {
            expect(RegistryClient).to.have.property('clearRuntimeCache');
        });
    });

    describe('instantiating a client with custom options', function () {
        describe('offline', function () {
            it('should not return search results if cache is empty', function (next) {
                // TODO: this test should be made individually for search, list and lookup
                this.registry.clearCache(function () {
                    this.registry._config.offline = true;
                    this.registry.search('jquery', function (err, results) {
                        expect(err).to.be(null);
                        expect(results.length).to.eql(0);
                        next();
                    });
                }.bind(this));
            });
        });

        describe('cache', function () {
            beforeEach(function () {
                nock('https://bower.herokuapp.com:443')
                  .get('/packages/search/jquery')
                  .replyWithFile(200, __dirname + '/fixtures/search.json');

                this.client = new RegistryClient({
                    cache: __dirname + '/cache',
                    strictSsl: false
                });

                this.cacheDir = this.client._config.cache;
                this.host = 'bower.herokuapp.com';
                this.method = 'search';
                this.pkg = 'jquery';

                this.path = this.cacheDir + '/' + this.host + '/' + this.method + '/' + this.pkg + '_' + md5(this.pkg).substr(0, 5);
            });

            afterEach(function (next) {
                this.client.clearCache(next);
            });

            it('should fill cache', function (next) {
                var self = this;

                // fill cache
                self.client.search(self.pkg, function (err, results) {
                    expect(err).to.be(null);
                    expect(results.length).to.eql(334);

                    // check for cache existence
                    fs.exists(self.path, function (exists) {
                        expect(exists).to.be(true);
                        next();
                    });
                });

            });

            it('should read results from cache', function (next) {
                var self = this;

                self.client.search(self.pkg, function (err, results) {
                    expect(err).to.be(null);
                    expect(results.length).to.eql(334);

                    fs.exists(self.path, function (exists) {
                        expect(exists).to.be(true);
                        next();
                    });
                });
            });
        });
    });


    //
    // lookup
    //
    describe('calling the lookup instance method with argument', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .get('/packages/jquery')
              .reply(200, {
                name: 'jquery',
                url: 'git://github.com/components/jquery.git'
            });

            this.registry._config.force = true;
        });

        it('should not return an error', function (next) {
            this.registry.lookup('jquery', function (err) {
                expect(err).to.be(null);
                next();
            });
        });

        it('should return entry type', function (next) {
            this.registry.lookup('jquery', function (err, entry) {
                expect(err).to.be(null);
                expect(entry.type).to.eql('alias');
                next();
            });
        });

        it('should return entry url ', function (next) {
            this.registry.lookup('jquery', function (err, entry) {
                expect(err).to.be(null);
                expect(entry.url).to.eql('git://github.com/components/jquery.git');
            });
            next();
        });
    });

    describe('calling the lookup instance method without argument', function () {
        it('should return no result', function (next) {
            this.timeout(10000);
            this.registry.lookup('', function (err, entry) {
                expect(err).to.not.be.ok();
                expect(entry).to.not.be.ok();
                next();
            });
        });
    });

    describe('calling the lookup instance method with two registries, and the first missing.', function () {
        beforeEach(function () {
            nock('http://custom-registry.com')
              .get('/packages/jquery')
              .reply(200, {
                'error': {
                    'message': 'missing',
                    'stack': 'Error: missing'
                }
            });

            nock('http://custom-registry2.com')
              .get('/packages/jquery')
              .reply(200, {
                name: 'jquery',
                url: 'git://github.com/foo/baz'
            });

            this.registry = new RegistryClient({
                strictSsl: false,
                force: true,
                registry: {
                    search: [
                        'http://custom-registry.com',
                        'http://custom-registry2.com'
                    ]
                }
            });
        });

        it('should return entry type', function (next) {
            this.registry.lookup('jquery', function (err, entry) {
                expect(err).to.be(null);
                expect(entry).to.be.an('object');
                expect(entry.type).to.eql('alias');
                next();
            });
        });

        it('should return entry url ', function (next) {
            this.registry.lookup('jquery', function (err, entry) {
                expect(err).to.be(null);
                expect(entry).to.be.an('object');
                expect(entry.url).to.eql('git://github.com/foo/baz');
                next();
            });
        });
    });

    describe('calling the lookup instance method with three registries', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .get('/packages/jquery')
              .reply(404);

            nock('http://custom-registry.com')
              .get('/packages/jquery')
              .reply(200, {
                name: 'jquery',
                url: 'git://github.com/foo/bar'
            });

            nock('http://custom-registry2.com')
              .get('/packages/jquery')
              .reply(200, {
                name: 'jquery',
                url: 'git://github.com/foo/baz'
            });

            this.registry = new RegistryClient({
                strictSsl: false,
                force: true,
                registry: {
                    search: [
                        'https://bower.herokuapp.com',
                        'http://custom-registry.com',
                        'http://custom-registry2.com'
                    ]
                }
            });
        });

        it('should return entry type', function (next) {
            this.registry.lookup('jquery', function (err, entry) {
                expect(err).to.be(null);
                expect(entry).to.be.an('object');
                expect(entry.type).to.eql('alias');
                next();
            });
        });

        it('should return entry url ', function (next) {
            this.registry.lookup('jquery', function (err, entry) {
                expect(err).to.be(null);
                expect(entry).to.be.an('object');
                expect(entry.url).to.eql('git://github.com/foo/bar');
                next();
            });
        });

        it('should respect order', function (next) {
            this.registry._config.registry.search = [
                'https://bower.herokuapp.com',
                'http://custom-registry2.com',
                'http://custom-registry.com'
            ];

            this.registry.lookup('jquery', function (err, entry) {
                expect(err).to.be(null);
                expect(entry).to.be.an('object');
                expect(entry.url).to.eql('git://github.com/foo/baz');
                next();
            });
        });
    });

    //
    // register
    //
    describe('calling the register instance method with argument', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .post('/packages', 'name=test-ba&url=git%3A%2F%2Fgithub.com%2Ftest-ba%2Ftest-ba.git')
              .reply(201);

            this.pkg = 'test-ba';
            this.pkgUrl = 'git://github.com/test-ba/test-ba.git';
        });

        it('should not return an error', function (next) {
            this.registry.register(this.pkg, this.pkgUrl, function (err) {
                expect(err).to.be(null);
                next();
            });
        });

        it('should return entry name', function (next) {
            var self = this;

            this.registry.register(this.pkg, this.pkgUrl, function (err, entry) {
                expect(err).to.be(null);
                expect(entry.name).to.eql(self.pkg);
                next();
            });
        });

        it('should return entry url', function (next) {
            var self = this;

            this.registry.register(this.pkg, this.pkgUrl, function (err, entry) {
                expect(err).to.be(null);
                expect(entry.url).to.eql(self.pkgUrl);
                next();
            });
        });
    });

    describe('calling the register instance method without arguments', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .post('/packages', 'name=&url=')
              .reply(400);
        });

        it('should return an error and no result', function (next) {
            this.registry.register('', '', function (err, entry) {
                expect(err).to.be.an(Error);
                expect(entry).to.be(undefined);
                next();
            });
        });
    });


    //
    // unregister
    //
    describe('calling the unregister instance method with argument', function () {
        beforeEach(function () {
            this.pkg = 'testfoo';
            this.accessToken = '12345678';
            this.registry._config.accessToken = this.accessToken;

            nock('https://bower.herokuapp.com:443')
              .delete('/packages/' + this.pkg + '?access_token=' + this.accessToken)
              .reply(204);
        });

        it('should not return an error when valid', function (next) {
            this.registry.unregister(this.pkg, function (err) {
                expect(err).to.be(null);
                next();
            });
        });

        it('should return entry name', function (next) {
            var self = this;

            this.registry.unregister(this.pkg, function (err, entry) {
                expect(err).to.be(null);
                expect(entry.name).to.eql(self.pkg);
                next();
            });
        });
    });

    describe('calling the unregister instance method with invalid token', function () {
        beforeEach(function () {
            this.pkg = 'testfoo';
            this.registry._config.accessToken = '';

            nock('https://bower.herokuapp.com:443')
              .delete('/packages/' + this.pkg)
              .reply(403);
        });

        it('should return an error', function (next) {
            this.registry.unregister(this.pkg, function (err, entry) {
                expect(err).to.be.an(Error);
                expect(entry).to.be(undefined);
                next();
            });
        });
    });

    describe('calling the unregister instance method with invalid package', function () {
        beforeEach(function () {
            this.notpkg = 'testbar';
            this.accessToken = '12345678';
            this.registry._config.accessToken = this.accessToken;

            nock('https://bower.herokuapp.com:443')
              .delete('/packages/' + this.notpkg + '?access_token=' + this.accessToken)
              .reply(404);
        });

        it('should return an error', function (next) {
            this.registry.unregister(this.notpkg, function (err, entry) {
                expect(err).to.be.an(Error);
                expect(entry).to.be(undefined);
                next();
            });
        });
    });

    //
    // search
    //
    describe('calling the search instance method with argument', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .get('/packages/search/jquery')
              .replyWithFile(200, __dirname + '/fixtures/search.json');

            this.pkg = 'jquery';
            this.pkgUrl = 'git://github.com/components/jquery.git';

            this.registry._config.force = true;
        });

        it('should not return an error', function (next) {
            this.registry.search(this.pkg, function (err) {
                expect(err).to.be(null);
                next();
            });
        });

        it('should return entry name', function (next) {
            var self = this;

            this.registry.search(this.pkg, function (err, results) {
                var found = results.some(function (entry) {
                    return entry.name === self.pkg;
                });

                expect(found).to.be(true);
                next();
            });
        });

        it('should return entry url', function (next) {
            var self = this;

            this.registry.search(this.pkg, function (err, results) {
                var found = results.some(function (entry) {
                    return entry.url === self.pkgUrl;
                });

                expect(found).to.be(true);
                next();
            });
        });
    });

    describe('calling the search instance method with two registries', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .get('/packages/search/jquery')
              .reply(200, []);

            nock('http://custom-registry.com')
              .get('/packages/search/jquery')
              .reply(200, [
                {
                    name: 'jquery',
                    url: 'git://github.com/bar/foo.git'
                }
            ]);

            this.pkg = 'jquery';
            this.pkgUrl = 'git://github.com/bar/foo.git';

            this.registry = new RegistryClient({
                strictSsl: false,
                force: true,
                registry: {
                    search: [
                        'https://bower.herokuapp.com',
                        'http://custom-registry.com'
                    ]
                }
            });
        });

        it('should return entry name', function (next) {
            var self = this;

            this.registry.search(this.pkg, function (err, results) {
                var found = results.some(function (entry) {
                    return entry.name === self.pkg;
                });

                expect(found).to.be(true);
                next();
            });
        });

        it('should return entry url', function (next) {
            var self = this;

            this.registry.search(this.pkg, function (err, results) {
                if (! results.length) {
                    return next(new Error('Result expected'));
                }

                var found = results.some(function (entry) {
                    return entry.url === self.pkgUrl;
                });

                expect(found).to.be(true);
                next();
            });
        });
    });

    describe('calling the search instance method without argument', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .get('/packages/search/')
              .reply(404);
        });

        it('should return an error and no results', function (next) {
            this.registry.search('', function (err, results) {
                expect(err).to.be.an(Error);
                expect(results).to.be(undefined);
                next();
            });
        });
    });

    //
    // list
    //
    describe('calling the list instance method', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .get('/packages')
              .reply(200, [], {});

            this.registry._config.force = true;
        });

        it('should not return an error', function (next) {
            this.registry.list(function (err) {
                expect(err).to.be(null);
                next();
            });
        });

        it('should return results array', function (next) {
            this.registry.list(function (err, results) {
                expect(results).to.be.an('array');
                next();
            });
        });

    });

    describe('calling the list instance method with two registries', function () {
        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .get('/packages')
              .reply(200, []);

            nock('http://custom-registry.com')
              .get('/packages')
              .reply(200, [
                {
                    name: 'jquery',
                    url: 'git://github.com/bar/foo.git'
                }
            ]);

            this.registry = new RegistryClient({
                strictSsl: false,
                force: true,
                registry: {
                    search: [
                        'https://bower.herokuapp.com',
                        'http://custom-registry.com'
                    ]
                }
            });
        });

        it('should return entry name', function (next) {
            var self = this;

            this.registry.list(function (err, results) {
                var found = results.some(function (entry) {
                    return entry.name === self.pkg;
                });

                expect(found).to.be(true);
                next();
            });
        });

        it('should return entry url', function (next) {
            var self = this;

            this.registry.list(function (err, results) {
                if (! results.length) {
                    return next(new Error('Result expected'));
                }

                var found = results.some(function (entry) {
                    return entry.url === self.pkgUrl;
                });

                expect(found).to.be(true);
                next();
            });
        });
    });

    describe('calling the list instance method', function () {

        beforeEach(function () {
            nock('https://bower.herokuapp.com:443')
              .get('/packages')
              .reply(200, [], {});
        });

        it('should return an error and no results', function (next) {
            this.registry.list(function (err) {
                expect(err).to.be(null);
                next();
            });
        });
    });

    //
    // clearCache
    //
    describe('called the clearCache instance method with argument', function () {
        beforeEach(function () {
            this.pkg = 'jquery';
        });

        it('should not return an error', function (next) {
            this.registry.clearCache(this.pkg, function (err) {
                expect(err).to.be(null);
                next();
            });
        });
    });

    describe('called the clearCache instance method without argument', function () {
        it('should not return any errors and remove all cache items', function (next) {
            this.registry.clearCache(function (err) {
                expect(err).to.be(null);
                next();
            });
        });
    });

    //
    // test userAgent
    //
    describe('add a custom userAgent with argument', function () {
        this.timeout(5000);
        it('should send custom userAgent to the server', function (next) {
            var self = this;
            this.ua = '';
            this.server = http.createServer(function (req, res) {
                self.ua = req.headers['user-agent'];
                res.writeHeader(200, {
                    'Content-Type': 'application/json'
                });
                res.end('{"name":"jquery","url":"git://github.com/components/jquery.git"}');
                self.server.close();
            });
            this.server.listen('7777', '127.0.0.1');
            this.registry = new RegistryClient({
                userAgent: 'test agent',
                registry: 'http://127.0.0.1:7777'
            });
            this.registry.search('jquery', function (err, result) {
                expect(self.ua).to.be('test agent');
                next();
            });
        });
    });
});
