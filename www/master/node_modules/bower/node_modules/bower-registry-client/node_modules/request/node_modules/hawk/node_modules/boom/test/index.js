// Load modules

var Lab = require('lab');
var Boom = require('../lib');


// Declare internals

var internals = {};


// Test shortcuts

var expect = Lab.expect;
var before = Lab.before;
var after = Lab.after;
var describe = Lab.experiment;
var it = Lab.test;


describe('Boom', function () {

    it('returns an error with info when constructed using another error', function (done) {

        var error = new Error('ka-boom');
        error.xyz = 123;
        var err = new Boom(error);
        expect(err.data.xyz).to.equal(123);
        expect(err.message).to.equal('ka-boom');
        expect(err.response).to.deep.equal({
            code: 500,
            payload: {
                code: 500,
                error: 'Internal Server Error',
                message: 'ka-boom'
            },
            headers: {}
        });
        done();
    });

    describe('#isBoom', function () {

        it('returns true for Boom object', function (done) {

            expect(Boom.badRequest().isBoom).to.equal(true);
            done();
        });

        it('returns false for Error object', function (done) {

            expect(new Error().isBoom).to.not.exist;
            done();
        });
    });

    describe('#badRequest', function () {

        it('returns a 400 error code', function (done) {

            expect(Boom.badRequest().response.code).to.equal(400);
            done();
        });

        it('sets the message with the passed in message', function (done) {

            expect(Boom.badRequest('my message').message).to.equal('my message');
            done();
        });
    });

    describe('#unauthorized', function () {

        it('returns a 401 error code', function (done) {

            var err = Boom.unauthorized();
            expect(err.response.code).to.equal(401);
            expect(err.response.headers).to.deep.equal({});
            done();
        });

        it('sets the message with the passed in message', function (done) {

            expect(Boom.unauthorized('my message').message).to.equal('my message');
            done();
        });

        it('returns a WWW-Authenticate header when passed a scheme', function (done) {

            var err = Boom.unauthorized('boom', 'Test');
            expect(err.response.code).to.equal(401);
            expect(err.response.headers['WWW-Authenticate']).to.equal('Test error="boom"');
            done();
        });

        it('returns a WWW-Authenticate header when passed a scheme and attributes', function (done) {

            var err = Boom.unauthorized('boom', 'Test', { a: 1, b: 'something', c: null, d: 0 });
            expect(err.response.code).to.equal(401);
            expect(err.response.headers['WWW-Authenticate']).to.equal('Test a="1", b="something", c="", d="0", error="boom"');
            done();
        });

        it('sets the isMissing flag when error message is empty', function (done) {

            var err = Boom.unauthorized('', 'Basic');
            expect(err.isMissing).to.equal(true);
            done();
        });

        it('does not set the isMissing flag when error message is not empty', function (done) {

            var err = Boom.unauthorized('message', 'Basic');
            expect(err.isMissing).to.equal(undefined);
            done();
        });

        it('sets a WWW-Authenticate when passed as an array', function (done) {

            var err = Boom.unauthorized('message', ['Basic', 'Example e="1"', 'Another x="3", y="4"']);
            expect(err.response.headers['WWW-Authenticate']).to.equal('Basic, Example e="1", Another x="3", y="4"');
            done();
        });
    });

    describe('#clientTimeout', function () {

        it('returns a 408 error code', function (done) {

            expect(Boom.clientTimeout().response.code).to.equal(408);
            done();
        });

        it('sets the message with the passed in message', function (done) {

            expect(Boom.clientTimeout('my message').message).to.equal('my message');
            done();
        });
    });

    describe('#serverTimeout', function () {

        it('returns a 503 error code', function (done) {

            expect(Boom.serverTimeout().response.code).to.equal(503);
            done();
        });

        it('sets the message with the passed in message', function (done) {

            expect(Boom.serverTimeout('my message').message).to.equal('my message');
            done();
        });
    });

    describe('#forbidden', function () {

        it('returns a 403 error code', function (done) {

            expect(Boom.forbidden().response.code).to.equal(403);
            done();
        });

        it('sets the message with the passed in message', function (done) {

            expect(Boom.forbidden('my message').message).to.equal('my message');
            done();
        });
    });

    describe('#notFound', function () {

        it('returns a 404 error code', function (done) {

            expect(Boom.notFound().response.code).to.equal(404);
            done();
        });

        it('sets the message with the passed in message', function (done) {

            expect(Boom.notFound('my message').message).to.equal('my message');
            done();
        });
    });

    describe('#internal', function () {

        it('returns a 500 error code', function (done) {

            expect(Boom.internal().response.code).to.equal(500);
            done();
        });

        it('sets the message with the passed in message', function (done) {

            var err = Boom.internal('my message');
            expect(err.message).to.equal('my message');
            expect(err.response.payload.message).to.equal('An internal server error occurred');
            done();
        });

        it('passes data on the callback if its passed in', function (done) {

            expect(Boom.internal('my message', { my: 'data' }).data.my).to.equal('data');
            done();
        });

        it('uses passed in stack if its available', function (done) {

            var error = new Error();
            error.stack = 'my stack line\nmy second stack line';
            expect(Boom.internal('my message', error).trace[0]).to.equal('my stack line');
            done();
        });
    });

    describe('#passThrough', function () {

        it('returns a pass-through error', function (done) {

            var err = Boom.passThrough(499, { a: 1 }, 'application/text', { 'X-Test': 'Boom' });
            expect(err.response.code).to.equal(499);
            expect(err.message).to.equal('Pass-through');
            expect(err.response).to.deep.equal({
                code: 499,
                payload: { a: 1 },
                headers: { 'X-Test': 'Boom' },
                type: 'application/text'
            });
            done();
        });
    });

    describe('#reformat', function () {

        it('encodes any HTML markup in the response payload', function (done) {

            var boom = new Boom(new Error('<script>alert(1)</script>'));
            expect(boom.response.payload.message).to.not.contain('<script>');
            done();
        });
    });
});


