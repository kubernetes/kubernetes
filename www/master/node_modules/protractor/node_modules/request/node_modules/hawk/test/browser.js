// Load modules

var Lab = require('lab');
var Hoek = require('hoek');
var Hawk = require('../lib');
var Browser = require('../lib/browser');
var LocalStorage = require('localStorage');


// Declare internals

var internals = {};


// Test shortcuts

var expect = Lab.expect;
var before = Lab.before;
var after = Lab.after;
var describe = Lab.experiment;
var it = Lab.test;


describe('Browser', function () {

    var credentialsFunc = function (id, callback) {

        var credentials = {
            id: id,
            key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
            algorithm: (id === '1' ? 'sha1' : 'sha256'),
            user: 'steve'
        };

        return callback(null, credentials);
    };

    it('should generate a header then successfully parse it (configuration)', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data' }).field;
            expect(req.authorization).to.exist;

            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                done();
            });
        });
    });

    it('should generate a header then successfully parse it (node request)', function (done) {

        var req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        var payload = 'some not so random text';

        credentialsFunc('123456', function (err, credentials) {

            var reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials, artifacts, req.headers['content-type'])).to.equal(true);

                var res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                res.headers['server-authorization'] = Hawk.server.header(credentials, artifacts, { payload: 'some reply', contentType: 'text/plain', ext: 'response-specific' });
                expect(res.headers['server-authorization']).to.exist;

                expect(Browser.client.authenticate(res, credentials, artifacts, { payload: 'some reply' })).to.equal(true);
                done();
            });
        });
    });

    it('should generate a header then successfully parse it (no server header options)', function (done) {

        var req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        var payload = 'some not so random text';

        credentialsFunc('123456', function (err, credentials) {

            var reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials, artifacts, req.headers['content-type'])).to.equal(true);

                var res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                res.headers['server-authorization'] = Hawk.server.header(credentials, artifacts);
                expect(res.headers['server-authorization']).to.exist;

                expect(Browser.client.authenticate(res, credentials, artifacts)).to.equal(true);
                done();
            });
        });
    });

    it('should generate a header then successfully parse it (no server header)', function (done) {

        var req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        var payload = 'some not so random text';

        credentialsFunc('123456', function (err, credentials) {

            var reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials, artifacts, req.headers['content-type'])).to.equal(true);

                var res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.client.authenticate(res, credentials, artifacts)).to.equal(true);
                done();
            });
        });
    });

    it('should generate a header with stale ts and successfully authenticate on second call', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {

            Browser.utils.setNtpOffset(60 * 60 * 1000);
            var header = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data' });
            req.authorization = header.field;
            expect(req.authorization).to.exist;

            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.exist;
                expect(err.message).to.equal('Stale timestamp');

                var res = {
                    headers: {
                        'www-authenticate': err.response.headers['WWW-Authenticate']
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.utils.getNtpOffset()).to.equal(60 * 60 * 1000);
                expect(Browser.client.authenticate(res, credentials, header.artifacts)).to.equal(true);
                expect(Browser.utils.getNtpOffset()).to.equal(0);

                req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data' }).field;
                expect(req.authorization).to.exist;

                Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                    expect(err).to.not.exist;
                    expect(credentials.user).to.equal('steve');
                    expect(artifacts.ext).to.equal('some-app-data');
                    done();
                });
            });
        });
    });

    it('should generate a header with stale ts and successfully authenticate on second call (manual localStorage)', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {
            Browser.utils.setStorage(LocalStorage)

            Browser.utils.setNtpOffset(60 * 60 * 1000);
            var header = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data' });
            req.authorization = header.field;
            expect(req.authorization).to.exist;

            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.exist;
                expect(err.message).to.equal('Stale timestamp');

                var res = {
                    headers: {
                        'www-authenticate': err.response.headers['WWW-Authenticate']
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(parseInt(LocalStorage.getItem('hawk_ntp_offset'))).to.equal(60 * 60 * 1000);
                expect(Browser.utils.getNtpOffset()).to.equal(60 * 60 * 1000);
                expect(Browser.client.authenticate(res, credentials, header.artifacts)).to.equal(true);
                expect(Browser.utils.getNtpOffset()).to.equal(0);
                expect(parseInt(LocalStorage.getItem('hawk_ntp_offset'))).to.equal(0);

                req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data' }).field;
                expect(req.authorization).to.exist;

                Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                    expect(err).to.not.exist;
                    expect(credentials.user).to.equal('steve');
                    expect(artifacts.ext).to.equal('some-app-data');
                    done();
                });
            });
        });
    });

    it('should generate a header then fails to parse it (missing server header hash)', function (done) {

        var req = {
            method: 'POST',
            url: '/resource/4?filter=a',
            headers: {
                host: 'example.com:8080',
                'content-type': 'text/plain;x=y'
            }
        };

        var payload = 'some not so random text';

        credentialsFunc('123456', function (err, credentials) {

            var reqHeader = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data', payload: payload, contentType: req.headers['content-type'] });
            req.headers.authorization = reqHeader.field;

            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload(payload, credentials, artifacts, req.headers['content-type'])).to.equal(true);

                var res = {
                    headers: {
                        'content-type': 'text/plain'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                res.headers['server-authorization'] = Hawk.server.header(credentials, artifacts);
                expect(res.headers['server-authorization']).to.exist;

                expect(Browser.client.authenticate(res, credentials, artifacts, { payload: 'some reply' })).to.equal(false);
                done();
            });
        });
    });

    it('should generate a header then successfully parse it (with hash)', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, payload: 'hola!', ext: 'some-app-data' }).field;
            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                done();
            });
        });
    });

    it('should generate a header then successfully parse it then validate payload', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, payload: 'hola!', ext: 'some-app-data' }).field;
            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(Hawk.server.authenticatePayload('hola!', credentials, artifacts)).to.be.true;
                expect(Hawk.server.authenticatePayload('hello!', credentials, artifacts)).to.be.false;
                done();
            });
        });
    });

    it('should generate a header then successfully parse it (app)', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data', app: 'asd23ased' }).field;
            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(artifacts.app).to.equal('asd23ased');
                done();
            });
        });
    });

    it('should generate a header then successfully parse it (app, dlg)', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data', app: 'asd23ased', dlg: '23434szr3q4d' }).field;
            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.not.exist;
                expect(credentials.user).to.equal('steve');
                expect(artifacts.ext).to.equal('some-app-data');
                expect(artifacts.app).to.equal('asd23ased');
                expect(artifacts.dlg).to.equal('23434szr3q4d');
                done();
            });
        });
    });

    it('should generate a header then fail authentication due to bad hash', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, payload: 'hola!', ext: 'some-app-data' }).field;
            Hawk.server.authenticate(req, credentialsFunc, { payload: 'byebye!' }, function (err, credentials, artifacts) {

                expect(err).to.exist;
                expect(err.response.payload.message).to.equal('Bad payload hash');
                done();
            });
        });
    });

    it('should generate a header for one resource then fail to authenticate another', function (done) {

        var req = {
            method: 'GET',
            url: '/resource/4?filter=a',
            host: 'example.com',
            port: 8080
        };

        credentialsFunc('123456', function (err, credentials) {

            req.authorization = Browser.client.header('http://example.com:8080/resource/4?filter=a', req.method, { credentials: credentials, ext: 'some-app-data' }).field;
            req.url = '/something/else';

            Hawk.server.authenticate(req, credentialsFunc, {}, function (err, credentials, artifacts) {

                expect(err).to.exist;
                expect(credentials).to.exist;
                done();
            });
        });
    });

    describe('client', function () {

        describe('#header', function () {

            it('should return a valid authorization header (sha1)', function (done) {

                var credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha1'
                };

                var header = Browser.client.header('http://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="bsvY3IfUllw6V5rvk4tStEvpBhE=", ext="Bazinga!", mac="qbf1ZPG/r/e06F4ht+T77LXi5vw="');
                done();
            });

            it('should return a valid authorization header (sha256)', function (done) {

                var credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                var header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", ext="Bazinga!", mac="q1CwFoSHzPZSkbIvl0oYlD+91rBUEvFk763nMjMndj8="');
                done();
            });

            it('should return a valid authorization header (no ext)', function (done) {

                var credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                var header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", mac="HTgtd0jPI6E4izx8e4OHdO36q00xFCU0FolNq3RiCYs="');
                done();
            });

            it('should return an empty authorization header on missing options', function (done) {

                var header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST').field;
                expect(header).to.equal('');
                done();
            });

            it('should return an empty authorization header on invalid credentials', function (done) {

                var credentials = {
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                var header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207 }).field;
                expect(header).to.equal('');
                done();
            });

            it('should return an empty authorization header on invalid algorithm', function (done) {

                var credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'hmac-sha-0'
                };

                var header = Browser.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, payload: 'something, anything!', ext: 'Bazinga!', timestamp: 1353809207 }).field;
                expect(header).to.equal('');
                done();
            });
        });

        describe('#authenticate', function () {

            it('should return false on invalid header', function (done) {

                var res = {
                    headers: {
                        'server-authorization': 'Hawk mac="abc", bad="xyz"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.client.authenticate(res, {})).to.equal(false);
                done();
            });

            it('should return false on invalid mac', function (done) {

                var res = {
                    headers: {
                        'content-type': 'text/plain',
                        'server-authorization': 'Hawk mac="_IJRsMl/4oL+nn+vKoeVZPdCHXB4yJkNnBbTbHFZUYE=", hash="f9cDF/TDm7TkYRLnGwRMfeDzT6LixQVLvrIKhh0vgmM=", ext="response-specific"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                var artifacts = {
                    method: 'POST',
                    host: 'example.com',
                    port: '8080',
                    resource: '/resource/4?filter=a',
                    ts: '1362336900',
                    nonce: 'eb5S_L',
                    hash: 'nJjkVtBE5Y/Bk38Aiokwn0jiJxt/0S2WRSUwWLCf5xk=',
                    ext: 'some-app-data',
                    app: undefined,
                    dlg: undefined,
                    mac: 'BlmSe8K+pbKIb6YsZCnt4E1GrYvY1AaYayNR82dGpIk=',
                    id: '123456'
                };

                var credentials = {
                    id: '123456',
                    key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
                    algorithm: 'sha256',
                    user: 'steve'
                };

                expect(Browser.client.authenticate(res, credentials, artifacts)).to.equal(false);
                done();
            });

            it('should return true on ignoring hash', function (done) {

                var res = {
                    headers: {
                        'content-type': 'text/plain',
                        'server-authorization': 'Hawk mac="XIJRsMl/4oL+nn+vKoeVZPdCHXB4yJkNnBbTbHFZUYE=", hash="f9cDF/TDm7TkYRLnGwRMfeDzT6LixQVLvrIKhh0vgmM=", ext="response-specific"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                var artifacts = {
                    method: 'POST',
                    host: 'example.com',
                    port: '8080',
                    resource: '/resource/4?filter=a',
                    ts: '1362336900',
                    nonce: 'eb5S_L',
                    hash: 'nJjkVtBE5Y/Bk38Aiokwn0jiJxt/0S2WRSUwWLCf5xk=',
                    ext: 'some-app-data',
                    app: undefined,
                    dlg: undefined,
                    mac: 'BlmSe8K+pbKIb6YsZCnt4E1GrYvY1AaYayNR82dGpIk=',
                    id: '123456'
                };

                var credentials = {
                    id: '123456',
                    key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
                    algorithm: 'sha256',
                    user: 'steve'
                };

                expect(Browser.client.authenticate(res, credentials, artifacts)).to.equal(true);
                done();
            });

            it('should fail on invalid WWW-Authenticate header format', function (done) {

                var res = {
                    headers: {
                        'www-authenticate': 'Hawk ts="1362346425875", tsm="PhwayS28vtnn3qbv0mqRBYSXebN/zggEtucfeZ620Zo=", x="Stale timestamp"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.client.authenticate(res, {})).to.equal(false);
                done();
            });

            it('should fail on invalid WWW-Authenticate header format', function (done) {

                var credentials = {
                    id: '123456',
                    key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
                    algorithm: 'sha256',
                    user: 'steve'
                };

                var res = {
                    headers: {
                        'www-authenticate': 'Hawk ts="1362346425875", tsm="hwayS28vtnn3qbv0mqRBYSXebN/zggEtucfeZ620Zo=", error="Stale timestamp"'
                    },
                    getResponseHeader: function (header) {

                        return res.headers[header.toLowerCase()];
                    }
                };

                expect(Browser.client.authenticate(res, credentials)).to.equal(false);
                done();
            });
        });

        describe('#message', function () {
            it('should generate an authorization then successfully parse it', function (done) {

                credentialsFunc('123456', function (err, credentials) {

                    var auth = Browser.client.message('example.com', 8080, 'some message', { credentials: credentials });
                    expect(auth).to.exist;

                    Hawk.server.authenticateMessage('example.com', 8080, 'some message', auth, credentialsFunc, {}, function (err, credentials) {

                        expect(err).to.not.exist;
                        expect(credentials.user).to.equal('steve');
                        done();
                    });
                });
            });

            it('should fail on missing host', function (done) {

                credentialsFunc('123456', function (err, credentials) {

                    var auth = Browser.client.message(null, 8080, 'some message', { credentials: credentials });
                    expect(auth).to.not.exist;
                    done();
                });
            });

            it('should fail on missing credentials', function (done) {

                var auth = Browser.client.message('example.com', 8080, 'some message', {});
                expect(auth).to.not.exist;
                done();
            });

            it('should fail on invalid algorithm', function (done) {

                credentialsFunc('123456', function (err, credentials) {

                    var creds = Hoek.clone(credentials);
                    creds.algorithm = 'blah';
                    var auth = Browser.client.message('example.com', 8080, 'some message', { credentials: creds });
                    expect(auth).to.not.exist;
                    done();
                });
            });
        });
    });

    describe('#parseAuthorizationHeader', function (done) {

        it('returns null on missing header', function (done) {

            expect(Browser.utils.parseAuthorizationHeader()).to.equal(null);
            done();
        });

        it('returns null on bad header syntax (structure)', function (done) {

            expect(Browser.utils.parseAuthorizationHeader('Hawk')).to.equal(null);
            done();
        });

        it('returns null on bad header syntax (parts)', function (done) {

            expect(Browser.utils.parseAuthorizationHeader(' ')).to.equal(null);
            done();
        });

        it('returns null on bad scheme name', function (done) {

            expect(Browser.utils.parseAuthorizationHeader('Basic asdasd')).to.equal(null);
            done();
        });

        it('returns null on bad attribute value', function (done) {

            expect(Browser.utils.parseAuthorizationHeader('Hawk test="\t"', ['test'])).to.equal(null);
            done();
        });

        it('returns null on duplicated attribute', function (done) {

            expect(Browser.utils.parseAuthorizationHeader('Hawk test="a", test="b"', ['test'])).to.equal(null);
            done();
        });
    });
});
