var lib = require('./lib/deap');

var deap = module.exports = lib.extendShallow;

deap(deap, {
	clone: lib.cloneShallow,
	extend: lib.extendShallow,
	update: lib.updateShallow,
	merge: lib.mergeShallow
});
