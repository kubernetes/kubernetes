'use strict';

var http = require('http');
var request = require('request');
var expect = require('expect.js');
var replay = require('../');

describe('request-replay', function () {
    it('should replay on network error', function (next) {
        replay(request.get('http://somedomainthatwillneverexistforsure.com:8089', function (error) {
            expect(error).to.be.an(Error);
            expect(error.code).to.equal('ENOTFOUND');
            expect(error.replays).to.equal(5);
            next();
        }), {
            errorCodes: ['ENOTFOUND'],
            factor: 1,
            minTimeout: 10,
            maxTimeout: 10
        });
    });

    it('should succeed if first fails but one of others succeed', function (next) {
        this.timeout(15000);

        replay(request.get('http://127.0.0.1:8089', { json: true }, function (error, response, body) {
            expect(error).to.be(null);
            expect(response).to.be.ok();
            expect(body).to.eql({ 'foo': 'bar' });
            next();
        }), {
            errorCodes: ['ENOTFOUND', 'ECONNREFUSED']
        })
        .on('replay', function () {
            http.createServer(function (req, res) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end('{ "foo": "bar" }');
            })
            .listen(8089, '127.0.0.1');
        });
    });

    it('should fire a replay event on each retry', function (next) {
        var stream;
        var tries = 0;

        stream = replay(request.get('http://somedomainthatwillneverexistforsure.com:8089', function (error) {
            expect(error).to.be.an(Error);
            expect(error.code).to.equal('ENOTFOUND');
            expect(error.replays).to.equal(5);
            next();
        }), {
            errorCodes: ['ENOTFOUND'],
            factor: 1,
            minTimeout: 10,
            maxTimeout: 10,
            randomize: false
        })
        .on('replay', function (replay) {
            expect(replay).to.be.an('object');
            expect(replay.number).to.equal(tries);
            expect(replay.error).to.be.an(Error);
            expect(replay.error.code).to.equal('ENOTFOUND');
            expect(replay.delay).to.be(10);
            tries++;
        });
    });
});
