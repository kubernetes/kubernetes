var lib = require('./lib/deap');

var deap = module.exports = lib.extend;

deap(deap, {
	clone: lib.clone,
	extend: lib.extend,
	update: lib.update,
	merge: lib.merge,
	cloneShallow: lib.cloneShallow,
	extendShallow: lib.extendShallow,
	updateShallow: lib.updateShallow,
	mergeShallow: lib.mergeShallow
});
