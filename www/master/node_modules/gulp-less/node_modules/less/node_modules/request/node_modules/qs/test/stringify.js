// Load modules

var Lab = require('lab');
var Qs = require('../');


// Declare internals

var internals = {};


// Test shortcuts

var expect = Lab.expect;
var before = Lab.before;
var after = Lab.after;
var describe = Lab.experiment;
var it = Lab.test;


describe('#stringify', function () {

    it('stringifies a querystring object', function (done) {

        expect(Qs.stringify({ a: 'b' })).to.equal('a=b');
        expect(Qs.stringify({ a: 1 })).to.equal('a=1');
        expect(Qs.stringify({ a: 1, b: 2 })).to.equal('a=1&b=2');
        done();
    });

    it('stringifies a nested object', function (done) {

        expect(Qs.stringify({ a: { b: 'c' } })).to.equal('a[b]=c');
        expect(Qs.stringify({ a: { b: { c: { d: 'e' } } } })).to.equal('a[b][c][d]=e');
        done();
    });

    it('stringifies an array value', function (done) {

        expect(Qs.stringify({ a: ['b', 'c', 'd'] })).to.equal('a[0]=b&a[1]=c&a[2]=d');
        done();
    });

    it('stringifies a nested array value', function (done) {

        expect(Qs.stringify({ a: { b: ['c', 'd'] } })).to.equal('a[b][0]=c&a[b][1]=d');
        done();
    });

    it('stringifies an object inside an array', function (done) {

        expect(Qs.stringify({ a: [{ b: 'c' }] })).to.equal('a[0][b]=c');
        expect(Qs.stringify({ a: [{ b: { c: [1] } }] })).to.equal('a[0][b][c][0]=1');
        done();
    });

    it('stringifies a complicated object', function (done) {

        expect(Qs.stringify({ a: { b: 'c', d: 'e' } })).to.equal('a[b]=c&a[d]=e');
        done();
    });

    it('stringifies an empty value', function (done) {

        expect(Qs.stringify({ a: '' })).to.equal('a=');
        expect(Qs.stringify({ a: '', b: '' })).to.equal('a=&b=');
        expect(Qs.stringify({ a: null })).to.equal('a');
        expect(Qs.stringify({ a: { b: null } })).to.equal('a[b]');
        done();
    });

    it('drops keys with a value of undefined', function (done) {

        expect(Qs.stringify({ a: undefined })).to.equal('');
        expect(Qs.stringify({ a: { b: undefined, c: null } })).to.equal('a[c]');
        done();
    });

    it('url encodes values', function (done) {

        expect(Qs.stringify({ a: 'b c' })).to.equal('a=b%20c');
        done();
    });

    it('stringifies a date', function (done) {

        var now = new Date();
        var str = 'a=' + encodeURIComponent(now.toISOString());
        expect(Qs.stringify({ a: now })).to.equal(str);
        done();
    });

    it('stringifies the weird object from qs', function (done) {

        expect(Qs.stringify({ 'my weird field': 'q1!2"\'w$5&7/z8)?' })).to.equal('my%20weird%20field=q1!2%22\'w%245%267%2Fz8)%3F');
        done();
    });

    it('skips properties that are part of the object prototype', function (done) {

        Object.prototype.crash = 'test';
        expect(Qs.stringify({ a: 'b'})).to.equal('a=b');
        expect(Qs.stringify({ a: { b: 'c' } })).to.equal('a[b]=c');
        delete Object.prototype.crash;
        done();
    });

    it('stringifies boolean values', function (done) {

        expect(Qs.stringify({ a: true })).to.equal('a=true');
        expect(Qs.stringify({ a: { b: true } })).to.equal('a[b]=true');
        expect(Qs.stringify({ b: false })).to.equal('b=false');
        expect(Qs.stringify({ b: { c: false } })).to.equal('b[c]=false');
        done();
    });

    it('stringifies buffer values', function (done) {

        expect(Qs.stringify({ a: new Buffer('test') })).to.equal('a=test');
        expect(Qs.stringify({ a: { b: new Buffer('test') } })).to.equal('a[b]=test');
        done();
    });
});
