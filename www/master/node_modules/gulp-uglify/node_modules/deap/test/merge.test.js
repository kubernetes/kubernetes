var lib = require('../lib/deap'),
	assert = require('chai').assert;

describe('shallow merge', function() {
	var shallowMerge = lib.mergeShallow;

	it('should merge everything into an empty object', function() {
		var a = { foo: 'bar' },
			result = shallowMerge({}, a);

		assert.deepEqual(result, a);
	});

	it('should return a reference to the first argument', function() {
		var a = { burp: 'adurp' },
			b = { burp: 'zing', grr: 'arghh' };

		var result = shallowMerge(a, b);

		assert.strictEqual(result, a);
	});

	it('should not replace existing values', function() {
		var a = { burp: 'adurp' },
			b = { burp: 'zing', grr: 'arghh' };

		var result = shallowMerge(a, b);

		assert.deepEqual(result, { burp: 'adurp', grr: 'arghh' });
		assert.equal(result.burp, a.burp);
	});

});

describe('deep merge', function() {
	var deepMerge = lib.merge;

	it('should return a reference to the first argument', function() {
		var a = { burp: 'adurp' },
			b = { burp: 'zing', grr: 'arghh' };

		var result = deepMerge(a, b);

		assert.strictEqual(result, a);
	});

	it('should merge a nested object one level deep', function() {
		var a = { foo: 'bar', deep: { foo: 'bar', baz: 'buzz' }},
			b = { foo: 'bop', deep: { foo: 'beep', biz: 'baz' } };

		var result = deepMerge(a, b);

		assert.equal(result.foo, 'bar');
		assert.equal(result.deep.foo, 'bar');
		assert.equal(result.deep.baz, 'buzz');
		assert.equal(result.deep.biz, 'baz');
	});

	it('should merge a nested object two levels deep', function() {
		var a = { foo: 'bar', deep: { hi: 'hello', deeper: { foo: 'bar', baz: 'buzz' }}},
			b = { foo: 'baz', deep: { hi: 'bye', bye: 'hi', deeper: { foo: 'beep', bop: 'boop' } } };

		var result = deepMerge({}, a, b);

		assert.equal(result.foo, a.foo);
		assert.isObject(result.deep);
		assert.equal(result.deep.hi, a.deep.hi);
		assert.equal(result.deep.bye, b.deep.bye);
		assert.isObject(result.deep.deeper);
		assert.equal(result.deep.deeper.foo, a.deep.deeper.foo);
		assert.equal(result.deep.deeper.baz, a.deep.deeper.baz);
		assert.equal(result.deep.deeper.bop, b.deep.deeper.bop);
	});

	it('should merge properties from multiple objects', function() {
		var a = { foo: ['one'], boo: 'far', poo: 'tar' },
			b = { foo: ['two', 'three'], zoo: 'car' },
			c = { boo: 'star', two: 'czar' };

		var result = deepMerge({}, a, b, c);

		assert.deepEqual(result, {
			foo: a.foo,
			boo: a.boo,
			poo: a.poo,
			zoo: b.zoo,
			two: c.two
		});
	});

	it('should not preserve nested object references', function() {
		var a = { foo: 'bar' },
			nested = { grr: 'argh' },
			newFoo = { burp: nested },
			b = { foo: newFoo, foo2: newFoo };

		var result = deepMerge(a, b);
		assert.equal(a.foo, 'bar');
		assert.deepEqual(a.foo2.burp, b.foo2.burp);
		assert.notStrictEqual(a.foo2.burp, nested);
	});

	it('should not override a string with an object', function() {
		var a = { foo: 'bar' },
			b = { foo: { biz: 'baz' } };

		var result = deepMerge(a, b);
		assert.deepEqual(a, { foo: 'bar' });
	});

	it('should preserve array references', function() {
		var a = { nested: [{ foo: 'bar' }] },
			b = { nested: [{ boo: 'far' }] },
			deep = a.nested;

		var result = deepMerge(a, b);

		assert.deepEqual(result.nested, a.nested);
		assert.notStrictEqual(result.nested, b.nested);
		assert.strictEqual(result.nested, deep);
	});

	it('should not preserve references in arrays', function() {
		var a = { nested: [{ foo: 'bar' }] },
			b = { nested: [{ boo: 'far' }] },
			deeper = a.nested[0];

		var result = deepMerge({}, a, b);

		assert.deepEqual(result.nested, a.nested);
		assert.notStrictEqual(result.nested[0], deeper);
	});

});
