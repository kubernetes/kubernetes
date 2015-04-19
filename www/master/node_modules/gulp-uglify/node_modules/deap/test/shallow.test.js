var assert = require('chai').assert,
	lib = require('../lib/deap'),
	shallow = require('../shallow');

describe('shallow', function() {

	it('should be defined correctly', function() {
		assert.isFunction(shallow);

		assert.isFunction(shallow.extend);
		assert.isFunction(shallow.update);
		assert.isFunction(shallow.merge);
		assert.isFunction(shallow.clone);
	});

	it('should have shallow functions', function() {
		assert.equal(shallow, lib.extendShallow);
		assert.equal(shallow.extend, lib.extendShallow);
		assert.equal(shallow.update, lib.updateShallow);
		assert.equal(shallow.merge, lib.mergeShallow);
		assert.equal(shallow.clone, lib.cloneShallow);
	});

});
