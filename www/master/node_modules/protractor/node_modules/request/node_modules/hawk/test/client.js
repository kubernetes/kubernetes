// Load modules

var Url = require('url');
var Lab = require('lab');
var Hawk = require('../lib');


// Declare internals

var internals = {};


// Test shortcuts

var expect = Lab.expect;
var before = Lab.before;
var after = Lab.after;
var describe = Lab.experiment;
var it = Lab.test;


describe('Hawk', function () {

    describe('client', function () {

        describe('#header', function () {

            it('should return a valid authorization header (sha1)', function (done) {

                var credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha1'
                };

                var header = Hawk.client.header('http://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="bsvY3IfUllw6V5rvk4tStEvpBhE=", ext="Bazinga!", mac="qbf1ZPG/r/e06F4ht+T77LXi5vw="');
                done();
            });

            it('should return a valid authorization header (sha256)', function (done) {

                var credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                var header = Hawk.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", ext="Bazinga!", mac="q1CwFoSHzPZSkbIvl0oYlD+91rBUEvFk763nMjMndj8="');
                done();
            });

            it('should return a valid authorization header (no ext)', function (done) {

                var credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                var header = Hawk.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, timestamp: 1353809207, nonce: 'Ygvqdz', payload: 'something to write about', contentType: 'text/plain' }).field;
                expect(header).to.equal('Hawk id="123456", ts="1353809207", nonce="Ygvqdz", hash="2QfCt3GuY9HQnHWyWD3wX68ZOKbynqlfYmuO2ZBRqtY=", mac="HTgtd0jPI6E4izx8e4OHdO36q00xFCU0FolNq3RiCYs="');
                done();
            });

            it('should return an empty authorization header on missing options', function (done) {

                var header = Hawk.client.header('https://example.net/somewhere/over/the/rainbow', 'POST').field;
                expect(header).to.equal('');
                done();
            });

            it('should return an empty authorization header on invalid credentials', function (done) {

                var credentials = {
                    key: '2983d45yun89q',
                    algorithm: 'sha256'
                };

                var header = Hawk.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, ext: 'Bazinga!', timestamp: 1353809207 }).field;
                expect(header).to.equal('');
                done();
            });

            it('should return an empty authorization header on invalid algorithm', function (done) {

                var credentials = {
                    id: '123456',
                    key: '2983d45yun89q',
                    algorithm: 'hmac-sha-0'
                };

                var header = Hawk.client.header('https://example.net/somewhere/over/the/rainbow', 'POST', { credentials: credentials, payload: 'something, anything!', ext: 'Bazinga!', timestamp: 1353809207 }).field;
                expect(header).to.equal('');
                done();
            });
        });

        describe('#authenticate', function () {

            it('should return false on invalid header', function (done) {

                var res = {
                    headers: {
                        'server-authorization': 'Hawk mac="abc", bad="xyz"'
                    }
                };

                expect(Hawk.client.authenticate(res, {})).to.equal(false);
                done();
            });

            it('should return false on invalid mac', function (done) {

                var res = {
                    headers: {
                        'content-type': 'text/plain',
                        'server-authorization': 'Hawk mac="_IJRsMl/4oL+nn+vKoeVZPdCHXB4yJkNnBbTbHFZUYE=", hash="f9cDF/TDm7TkYRLnGwRMfeDzT6LixQVLvrIKhh0vgmM=", ext="response-specific"'
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

                expect(Hawk.client.authenticate(res, credentials, artifacts)).to.equal(false);
                done();
            });

            it('should return true on ignoring hash', function (done) {

                var res = {
                    headers: {
                        'content-type': 'text/plain',
                        'server-authorization': 'Hawk mac="XIJRsMl/4oL+nn+vKoeVZPdCHXB4yJkNnBbTbHFZUYE=", hash="f9cDF/TDm7TkYRLnGwRMfeDzT6LixQVLvrIKhh0vgmM=", ext="response-specific"'
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

                expect(Hawk.client.authenticate(res, credentials, artifacts)).to.equal(true);
                done();
            });

            it('should fail on invalid WWW-Authenticate header format', function (done) {

                var header = 'Hawk ts="1362346425875", tsm="PhwayS28vtnn3qbv0mqRBYSXebN/zggEtucfeZ620Zo=", x="Stale timestamp"';
                expect(Hawk.client.authenticate({ headers: { 'www-authenticate': header } }, {})).to.equal(false);
                done();
            });

            it('should fail on invalid WWW-Authenticate header format', function (done) {

                var credentials = {
                    id: '123456',
                    key: 'werxhqb98rpaxn39848xrunpaw3489ruxnpa98w4rxn',
                    algorithm: 'sha256',
                    user: 'steve'
                };

                var header = 'Hawk ts="1362346425875", tsm="hwayS28vtnn3qbv0mqRBYSXebN/zggEtucfeZ620Zo=", error="Stale timestamp"';
                expect(Hawk.client.authenticate({ headers: { 'www-authenticate': header } }, credentials)).to.equal(false);
                done();
            });
        });
    });
});
