var assert = require('chai').assert,
	lib = require('../lib/deap');

describe('shallow extend', function() {
	var shallow = lib.extendShallow;

	it('should return a reference to the first argument', function() {
		var a = { burp: 'adurp' },
			b = { burp: 'zing', grr: 'arghh' };

		var result = shallow(a, b);

		assert.strictEqual(result, a);
	});

	it('should copy simple values', function() {
		var a = {},
			b = { s: 'string', n: 1, b: false, a: [], o: {}};

		var c = shallow(a, b);

		assert.deepEqual(c, a);
		assert.equal(c.s, b.s);
		assert.equal(c.n, b.n);
		assert.equal(c.b, b.b);
		assert.strictEqual(c.a, b.a);
		assert.strictEqual(c.o, b.o);
	});

	it('should only alter first param', function() {
		var a = { doom: 'song' },
			b = { burp: 'adurp' },
			c = { grr: 'argh' };

		var result = shallow({}, a, b, c);

		assert.deepEqual(a, { doom: 'song' });
		assert.deepEqual(b, { burp: 'adurp' });
		assert.deepEqual(c, { grr: 'argh' });
		assert.sameMembers(Object.keys(result), ['doom', 'burp', 'grr']);
		assert.equal(result.doom, a.doom);
		assert.equal(result.burp, b.burp);
		assert.equal(result.grr, c.grr);

	});

	it('should preserve object references', function() {
		var deep = { foo: 'bar' },
			a = { burp: 'adurp' , nested: deep };


		var result = shallow({}, a);

		assert.strictEqual(result.nested, deep);
	});

	it('should preserve date references', function() {
		var a = { burp: 'adurp', date: new Date() },
			date = a.date;

		var result = shallow({}, a);

		assert.strictEqual(result.date, date);
	});

	it('should preserve regexp references', function() {
		var a = { burp: 'adurp', regexp: /foo/g },
			regexp = a.regexp;

		var result = shallow({}, a);

		assert.strictEqual(result.regexp, regexp);
	});

	it('should preserve array references', function() {
		var a = { burp: 'adurp', array: [] },
			array = a.array;

		var result = shallow({}, a);

		assert.strictEqual(result.array, array);
	});

	it('should not pick up non-enumberable properties', function() {
		var result = shallow({}, function() {});

		assert.deepEqual(result, {});
		assert.equal(Object.keys(result).length, 0);
		assert.equal(Object.getOwnPropertyNames(result).length, 0);
	});
});

describe('deep extend', function() {
	var deepExtend = lib.extend;

	it('should return a reference to the first argument', function() {
		var a = { burp: 'adurp' },
			b = { burp: 'zing', grr: 'arghh' };

		var result = deepExtend(a, b);

		assert.strictEqual(result, a);
	});

	it('should not preserve object references', function() {
		var deeper = { boo: 'far' },
			deep = { foo: 'bar', nested: deeper },
			a = { burp: 'adurp' , nested: deep };

		var result = deepExtend({}, a);

		assert.deepEqual(result, a);
		assert.notStrictEqual(result.nested, deep);
		assert.notStrictEqual(result.nested.nested, deeper);
	});

	it('should not preserve date references', function() {
		var a = { burp: 'adurp', date: new Date() },
			date = a.date;

		var result = deepExtend({}, a);

		assert.deepEqual(result, a);
		assert.equal(result.date.getTime(), date.getTime()); // added this because deepEqual doesn't work with dates
		assert.notStrictEqual(result.date, date);
	});

	it('should not preserve regexp references', function() {
		var a = { burp: 'adurp', regexp: /foo/g },
			regexp = a.regexp;

		var result = deepExtend({}, a);

		assert.deepEqual(result, a);
		assert.notStrictEqual(result.regexp, regexp);
	});

	it('should not preserve array references', function() {
		var deeper = { boo: 'far' },
			deep = { foo: 'bar', nested: deeper },
			a = { burp: 'adurp' , nested: [deep, deeper] };

		var result = deepExtend({}, a);

		assert.deepEqual(result, a);
		assert.notStrictEqual(result.nested, a.nested);
		assert.notStrictEqual(result.nested[0], deep);
		assert.notStrictEqual(result.nested[0].nested, deeper);
		assert.notStrictEqual(result.nested[1], deeper);

		assert.deepEqual(result.nested[0].nested, result.nested[1]);
		assert.notStrictEqual(result.nested[0].nested, result.nested[1]);
	});

});
