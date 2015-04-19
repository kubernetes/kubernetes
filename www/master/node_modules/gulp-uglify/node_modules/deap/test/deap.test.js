var assert = require('chai').assert,
	deap = require('../index'),
	lib = require('../lib/deap');

describe('deap', function() {

	it('should have clone defined', function() {
		assert.isFunction(deap.clone);
		assert.deepEqual(deap.clone, lib.clone);
	});

	it('should have cloneShallow defined', function() {
		assert.isFunction(deap.cloneShallow);
		assert.deepEqual(deap.cloneShallow, lib.cloneShallow);
	});

	it('should have extend exposed as a top level function', function() {
		assert.isFunction(deap);
		assert.equal(deap, lib.extend);
	});

	it('should have extend defined', function() {
		assert.isFunction(deap.extend);
		assert.deepEqual(deap.extend, lib.extend);
	});

	it('should have extendShallow defined', function() {
		assert.isFunction(deap.extendShallow);
		assert.deepEqual(deap.extendShallow, lib.extendShallow);
	});

	it('should have update defined', function() {
		assert.isFunction(deap.update);
		assert.deepEqual(deap.update, lib.update);
	});

	it('should have updateShallow defined', function() {
		assert.isFunction(deap.updateShallow);
		assert.deepEqual(deap.updateShallow, lib.updateShallow);
	});

	it('should have merge defined', function() {
		assert.isFunction(deap.merge);
		assert.deepEqual(deap.merge, lib.merge);
	});

	it('should have mergeShallow defined', function() {
		assert.isFunction(deap.mergeShallow);
		assert.deepEqual(deap.mergeShallow, lib.mergeShallow);
	});


});
