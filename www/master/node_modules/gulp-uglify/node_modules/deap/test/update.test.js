var lib = require('../lib/deap'),
	assert = require('chai').assert;

describe('shallow update', function() {
	var shallowUpdate = lib.updateShallow;

	it('should not update anything into an empty object', function() {
		var result = shallowUpdate({}, { foo: 'bar' });

		assert.deepEqual(result, {});
	});

	it('should return a reference to the first argument', function() {
		var a = { burp: 'adurp' },
			b = { burp: 'zing', grr: 'arghh' };

		var result = shallowUpdate(a, b);

		assert.strictEqual(result, a);
	});

	it('should replace existing values only', function() {
		var a = { burp: 'adurp' },
			b = { burp: 'zing', grr: 'arghh' };

		var result = shallowUpdate(a, b);

		assert.deepEqual(result, a);
		assert.equal(a.burp, b.burp);
		assert.isUndefined(a.grr);
	});

});

describe('deep update', function() {
	var deepUpdate = lib.update;

	it('should return a reference to the first argument', function() {
		var a = { burp: 'adurp' },
			b = { burp: 'zing', grr: 'arghh' };

		var result = deepUpdate(a, b);

		assert.strictEqual(result, a);
	});

	it('should update a nested object one level deep', function() {
		var a = { foo: 'bar', deep: { foo: 'bar', baz: 'buzz' }},
			b = { deep: { foo: 'beep' } };

		var result = deepUpdate(a, b);

		assert.equal(result.foo, a.foo);
		assert.equal(result.deep.foo, b.deep.foo);
		assert.equal(result.deep.baz, a.deep.baz);
	});

	it('should update a nested object two levels deep', function() {
		var a = { foo: 'bar', deep: { hi: 'hello', deeper: { foo: 'bar', baz: 'buzz' }}},
			b = { deep: { deeper: { foo: 'beep' } } };

		var result = deepUpdate(a, b);

		assert.equal(result.foo, a.foo);
		assert.isObject(result.deep);
		assert.equal(result.deep.hi, a.deep.hi);
		assert.isObject(result.deep.deeper);
		assert.equal(result.deep.deeper.foo, b.deep.deeper.foo);
		assert.equal(result.deep.deeper.baz, a.deep.deeper.baz);
	});

	it('should update properties from multiple objects', function() {
		var a = { foo: ['one'], boo: 'far', poo: 'tar' },
			b = { foo: ['two', 'three'], zoo: 'car' },
			c = { boo: 'star', two: 'czar' };

		var result = deepUpdate(a, b, c);

		assert.deepEqual(result, {
			foo: b.foo,
			boo: c.boo,
			poo: a.poo
		});
	});

	it('should not update properties that are not on the first argument', function() {
		var a = { foo: 'bar', deep: { deeper: { foo: 'bar' } } },
			b = { boo: 'far', deep: { hi: 'hello', deeper: { foo: 'beep', baz: 'buzz' } } };

		var result = deepUpdate(a, b);

		assert.isUndefined(result.boo);
		assert.isObject(result.deep);
		assert.isUndefined(result.deep.hi);
		assert.isObject(result.deep.deeper);
		assert.isUndefined(result.deep.deeper.baz);
		assert.equal(result.deep.deeper.foo, b.deep.deeper.foo);
	});

	it('should not preserve nested object references', function() {
		var a = { foo: 'bar' },
			nested = { grr: 'argh' },
			newFoo = { burp: nested },
			b = { foo: newFoo };

		var result = deepUpdate(a, b);

		assert.deepEqual(a.foo.burp, b.foo.burp);
		assert.notStrictEqual(a.foo.burp, nested);
	});

	it('should preserve array references', function() {
		var a = { nested: [{ foo: 'bar' }] },
			b = { nested: [{ boo: 'far' }] },
			deep = a.nested;

		var result = deepUpdate(a, b);

		assert.deepEqual(result.nested, b.nested);
		assert.notStrictEqual(result.nested, b.nested);
		assert.strictEqual(result.nested, a.nested);
		assert.strictEqual(result.nested, deep);
	});

	it('should not preserve references in arrays', function() {
		var a = { nested: [{ foo: 'bar' }] },
			b = { nested: [{ boo: 'far' }] },
			deeper = a.nested[0];

		var result = deepUpdate(a, b);

		assert.deepEqual(result.nested, b.nested);
		assert.notStrictEqual(result.nested[0], deeper);
	});

});
