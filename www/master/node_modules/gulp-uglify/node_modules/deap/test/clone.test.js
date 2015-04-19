var lib = require('../lib/deap'),
	assert = require('chai').assert;

describe('shallow clone', function() {
	var shallow = lib.cloneShallow;

	it('should not return a reference to the first argument', function() {
		var a = { burp: 'adurp' },
			result = shallow(a);

		assert.notStrictEqual(result, a);
	});

	it('should copy simple values', function() {
		var a = { s: 'string', n: 1, b: false, a: [], o: {}},
			b = shallow(a);

		assert.deepEqual(b, a);
		assert.equal(b.s, a.s);
		assert.equal(b.n, a.n);
		assert.equal(b.b, a.b);
		assert.strictEqual(b.a, a.a);
		assert.strictEqual(b.o, a.o);
	});

	it('should preserve object references', function() {
		var deep = { foo: 'bar' },
			a = { burp: 'adurp' , nested: deep },
			result = shallow(a);

		assert.strictEqual(result.nested, deep);
	});

	it('should preserve date references', function() {
		var a = { burp: 'adurp', date: new Date() },
			date = a.date;

		var result = shallow(a);

		assert.strictEqual(result.date, date);
	});

	it('should preserve regexp references', function() {
		var a = { burp: 'adurp', regexp: /foo/g },
			regexp = a.regexp;

		var result = shallow(a);

		assert.strictEqual(result.regexp, regexp);
	});

	it('should preserve array references', function() {
		var a = { burp: 'adurp', array: [] },
			array = a.array;

		var result = shallow(a);

		assert.strictEqual(result.array, array);
	});

	it('should clone Date objects', function() {
		var a = new Date();

		var result = shallow(a);

		assert.equal(result.toString(), a.toString());
		assert.notStrictEqual(result, a);
	});

	it('should clone RegExp objects', function() {
		var a = /foo/;

		var result = shallow(a);

		assert.equal(result.toString(), a.toString());
		assert.notStrictEqual(result, a);
	});

	it('should work for multiple arguments', function() {
		var a = { doom: 'song' },
			b = { burp: 'adurp' },
			c = { grr: { doh: 'argh' } };

		var result = shallow(a, b, c);

		assert.deepEqual(a, { doom: 'song' });
		assert.deepEqual(b, { burp: 'adurp' });
		assert.deepEqual(c, { grr: { doh: 'argh' } });
		assert.sameMembers(Object.keys(result), ['doom', 'burp', 'grr']);
		assert.equal(result.doom, a.doom);
		assert.equal(result.burp, b.burp);
		assert.deepEqual(result.grr, c.grr);
		assert.strictEqual(result.grr, c.grr);
	});

	describe('on an array', function() {

		it('should preserve references', function() {
			var a = ['string', 1, false, [], {}];

			var result = shallow(a);

			assert.deepEqual(result, a);
			assert.equal(result[0], a[0]);
			assert.equal(result[1], a[1]);
			assert.equal(result[2], a[2]);
			assert.strictEqual(result[3], a[3]);
			assert.strictEqual(result[4], a[4]);
		});

	});

});


describe('clone', function() {
	var clone = lib.clone;

	it('should not return a reference to the first argument', function() {
		var a = { burp: 'adurp' },
			result = clone(a);

		assert.notStrictEqual(result, a);
	});

	it('should copy simple values', function() {
		var a = { s: 'string', n: 1, b: false, a: [], o: {}},
			b = clone(a);

		assert.deepEqual(b, a);
		assert.equal(b.s, a.s);
		assert.equal(b.n, a.n);
		assert.equal(b.b, a.b);
		assert.deepEqual(b.a, a.a);
		assert.deepEqual(b.o, a.o);
	});

	it('should not preserve object references', function() {
		var deeper = { boo: 'far' },
			deep = { foo: 'bar', nested: deeper },
			a = { burp: 'adurp' , nested: deep };

		var result = clone(a);

		assert.deepEqual(result, a);
		assert.notStrictEqual(result.nested, deep);
		assert.notStrictEqual(result.nested.nested, deeper);
	});

	it('should not preserve date references', function() {
		var a = { burp: 'adurp', date: new Date() },
			date = a.date;

		var result = clone(a);

		assert.deepEqual(result, a);
		assert.equal(result.date.getTime(), date.getTime()); // added this because deepEqual doesn't work with dates
		assert.notStrictEqual(result.date, date);
	});

	it('should not preserve regexp references', function() {
		var a = { burp: 'adurp', regexp: /foo/g },
			regexp = a.regexp;

		var result = clone(a);

		assert.deepEqual(result, a);
		assert.notStrictEqual(result.regexp, regexp);
	});

	it('should not preserve array references', function() {
		var deeper = { boo: 'far' },
			deep = { foo: 'bar', nested: deeper },
			a = { burp: 'adurp' , nested: [deep, deeper] };

		var result = clone(a);

		assert.deepEqual(result, a);
		assert.notStrictEqual(result.nested, a.nested);
		assert.notStrictEqual(result.nested[0], deep);
		assert.notStrictEqual(result.nested[0].nested, deeper);
		assert.notStrictEqual(result.nested[1], deeper);

		assert.deepEqual(result.nested[0].nested, result.nested[1]);
		assert.notStrictEqual(result.nested[0].nested, result.nested[1]);
	});

	it('should clone Date objects', function() {
		var a = new Date();

		var result = clone(a);

		assert.equal(result.toString(), a.toString());
		assert.notStrictEqual(result, a);
	});

	it('should clone RegExp objects', function() {
		var a = /foo/;

		var result = clone(a);

		assert.equal(result.toString(), a.toString());
		assert.notStrictEqual(result, a);
	});

	it('should work for multiple arguments', function() {
		var a = { doom: 'song' },
			b = { burp: 'adurp' },
			c = { grr: { doh: 'argh' } };

		var result = clone(a, b, c);

		assert.deepEqual(a, { doom: 'song' });
		assert.deepEqual(b, { burp: 'adurp' });
		assert.deepEqual(c, { grr: { doh: 'argh' } });
		assert.sameMembers(Object.keys(result), ['doom', 'burp', 'grr']);
		assert.equal(result.doom, a.doom);
		assert.equal(result.burp, b.burp);
		assert.deepEqual(result.grr, c.grr);
		assert.notStrictEqual(result.grr, c.grr);
	});

	describe('on an array', function() {

		it('should not preserve references', function() {
			var a = ['string', 1, false, [], {}];

			var result = clone(a);

			assert.deepEqual(result, a);
			assert.equal(result[0], a[0]);
			assert.equal(result[1], a[1]);
			assert.equal(result[2], a[2]);
			assert.notStrictEqual(result[3], a[3]);
			assert.notStrictEqual(result[4], a[4]);
		});

	});

});
