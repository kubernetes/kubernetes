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


describe('#parse', function () {

    it('parses a simple string', function (done) {

        expect(Qs.parse('0=foo')).to.deep.equal({ '0': 'foo' });
        expect(Qs.parse('foo=c++')).to.deep.equal({ foo: 'c  ' });
        expect(Qs.parse('a[>=]=23')).to.deep.equal({ a: { '>=': '23' } });
        expect(Qs.parse('a[<=>]==23')).to.deep.equal({ a: { '<=>': '=23' } });
        expect(Qs.parse('a[==]=23')).to.deep.equal({ a: { '==': '23' } });
        expect(Qs.parse('foo')).to.deep.equal({ foo: '' });
        expect(Qs.parse('foo=bar')).to.deep.equal({ foo: 'bar' });
        expect(Qs.parse(' foo = bar = baz ')).to.deep.equal({ ' foo ': ' bar = baz ' });
        expect(Qs.parse('foo=bar=baz')).to.deep.equal({ foo: 'bar=baz' });
        expect(Qs.parse('foo=bar&bar=baz')).to.deep.equal({ foo: 'bar', bar: 'baz' });
        expect(Qs.parse('foo=bar&baz')).to.deep.equal({ foo: 'bar', baz: '' });
        expect(Qs.parse('cht=p3&chd=t:60,40&chs=250x100&chl=Hello|World')).to.deep.equal({
            cht: 'p3',
            chd: 't:60,40',
            chs: '250x100',
            chl: 'Hello|World'
        });
        done();
    });

    it('parses a single nested string', function (done) {

        expect(Qs.parse('a[b]=c')).to.deep.equal({ a: { b: 'c' } });
        done();
    });

    it('parses a double nested string', function (done) {

        expect(Qs.parse('a[b][c]=d')).to.deep.equal({ a: { b: { c: 'd' } } });
        done();
    });

    it('defaults to a depth of 5', function (done) {

        expect(Qs.parse('a[b][c][d][e][f][g][h]=i')).to.deep.equal({ a: { b: { c: { d: { e: { f: { '[g][h]': 'i' } } } } } } });
        done();
    });

    it('only parses one level when depth = 1', function (done) {

        expect(Qs.parse('a[b][c]=d', 1)).to.deep.equal({ a: { b: { '[c]': 'd' } } });
        expect(Qs.parse('a[b][c][d]=e', 1)).to.deep.equal({ a: { b: { '[c][d]': 'e' } } });
        done();
    });

    it('parses a simple array', function (done) {

        expect(Qs.parse('a=b&a=c')).to.deep.equal({ a: ['b', 'c'] });
        done();
    });

    it('parses an explicit array', function (done) {

        expect(Qs.parse('a[]=b')).to.deep.equal({ a: ['b'] });
        expect(Qs.parse('a[]=b&a[]=c')).to.deep.equal({ a: ['b', 'c'] });
        expect(Qs.parse('a[]=b&a[]=c&a[]=d')).to.deep.equal({ a: ['b', 'c', 'd'] });
        done();
    });

    it('parses a nested array', function (done) {

        expect(Qs.parse('a[b][]=c&a[b][]=d')).to.deep.equal({ a: { b: ['c', 'd'] } });
        expect(Qs.parse('a[>=]=25')).to.deep.equal({ a: { '>=': '25' } });
        done();
    });

    it('allows to specify array indices', function (done) {

        expect(Qs.parse('a[1]=c&a[0]=b&a[2]=d')).to.deep.equal({ a: ['b', 'c', 'd'] });
        expect(Qs.parse('a[1]=c&a[0]=b')).to.deep.equal({ a: ['b', 'c'] });
        expect(Qs.parse('a[1]=c')).to.deep.equal({ a: ['c'] });
        done();
    });

    it('limits specific array indices to 20', function (done) {

        expect(Qs.parse('a[20]=a')).to.deep.equal({ a: ['a'] });
        expect(Qs.parse('a[21]=a')).to.deep.equal({ a: { '21': 'a' } });
        done();
    });

    it('supports encoded = signs', function (done) {

        expect(Qs.parse('he%3Dllo=th%3Dere')).to.deep.equal({ 'he=llo': 'th=ere' });
        done();
    });

    it('is ok with url encoded strings', function (done) {

        expect(Qs.parse('a[b%20c]=d')).to.deep.equal({ a: { 'b c': 'd' } });
        expect(Qs.parse('a[b]=c%20d')).to.deep.equal({ a: { b: 'c d' } });
        done();
    });

    it('allows brackets in the value', function (done) {

        expect(Qs.parse('pets=["tobi"]')).to.deep.equal({ pets: '["tobi"]' });
        expect(Qs.parse('operators=[">=", "<="]')).to.deep.equal({ operators: '[">=", "<="]' });
        done();
    });

    it('allows empty values', function (done) {

        expect(Qs.parse('')).to.deep.equal({});
        expect(Qs.parse(null)).to.deep.equal({});
        expect(Qs.parse(undefined)).to.deep.equal({});
        done();
    });

    it('transforms arrays to objects', function (done) {

        expect(Qs.parse('foo[0]=bar&foo[bad]=baz')).to.deep.equal({ foo: { '0': 'bar', bad: 'baz' } });
        expect(Qs.parse('foo[bad]=baz&foo[0]=bar')).to.deep.equal({ foo: { bad: 'baz', '0': 'bar' } });
        expect(Qs.parse('foo[bad]=baz&foo[]=bar')).to.deep.equal({ foo: { bad: 'baz', '0': 'bar' } });
        expect(Qs.parse('foo[]=bar&foo[bad]=baz')).to.deep.equal({ foo: { '0': 'bar', bad: 'baz' } });
        expect(Qs.parse('foo[bad]=baz&foo[]=bar&foo[]=foo')).to.deep.equal({ foo: { bad: 'baz', '0': 'bar', '1': 'foo' } });
        done();
    });

    it('correctly prunes undefined values when converting an array to an object', function (done) {

        expect(Qs.parse('a[2]=b&a[99999999]=c')).to.deep.equal({ a: { '2': 'b', '99999999': 'c' } });
        done();
    });

    it('supports malformed uri characters', function (done) {

        expect(Qs.parse('{%:%}')).to.deep.equal({ '{%:%}': '' });
        expect(Qs.parse('foo=%:%}')).to.deep.equal({ foo: '%:%}' });
        done();
    });

    it('doesn\'t produce empty keys', function (done) {

        expect(Qs.parse('_r=1&')).to.deep.equal({ '_r': '1' });
        done();
    });

    it('cannot override prototypes', function (done) {

        var obj = Qs.parse('toString=bad&bad[toString]=bad&constructor=bad');
        expect(typeof obj.toString).to.equal('function');
        expect(typeof obj.bad.toString).to.equal('function');
        expect(typeof obj.constructor).to.equal('function');
        done();
    });

    it('cannot access Object prototype', function (done) {

        Qs.parse('constructor[prototype][bad]=bad');
        Qs.parse('bad[constructor][prototype][bad]=bad');
        expect(typeof Object.prototype.bad).to.equal('undefined');
        done();
    });

    it('parses arrays of objects', function (done) {

        expect(Qs.parse('a[][b]=c')).to.deep.equal({ a: [{ b: 'c' }] });
        expect(Qs.parse('a[0][b]=c')).to.deep.equal({ a: [{ b: 'c' }] });
        done();
    });

    it('should compact sparse arrays', function (done) {

        expect(Qs.parse('a[10]=1&a[2]=2')).to.deep.equal({ a: ['2', '1'] });
        done();
    });

    it('parses semi-parsed strings', function (done) {

        expect(Qs.parse({ 'a[b]': 'c' })).to.deep.equal({ a: { b: 'c' } });
        expect(Qs.parse({ 'a[b]': 'c', 'a[d]': 'e' })).to.deep.equal({ a: { b: 'c', d: 'e' } });
        done();
    });

    it('parses buffers to strings', function (done) {

        var b = new Buffer('test');
        expect(Qs.parse({ a: b })).to.deep.equal({ a: b.toString() });
        done();
    });

    it('continues parsing when no parent is found', function (done) {

        expect(Qs.parse('[]&a=b')).to.deep.equal({ '0': '', a: 'b' });
        expect(Qs.parse('[foo]=bar')).to.deep.equal({ foo: 'bar' });
        done();
    });

    it('does not error when parsing a very long array', function (done) {

        var str = 'a[]=a';
        while (Buffer.byteLength(str) < 128 * 1024) {
            str += '&' + str;
        }

        expect(function () {

            Qs.parse(str);
        }).to.not.throw();

        done();
    });

    it('should not throw when a native prototype has an enumerable property', { parallel: false }, function (done) {

        Object.prototype.crash = '';
        expect(Qs.parse.bind(null, 'test')).to.not.throw();
        delete Object.prototype.crash;
        done();
    });
});
