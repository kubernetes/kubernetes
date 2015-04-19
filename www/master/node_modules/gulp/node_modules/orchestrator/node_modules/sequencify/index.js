/*jshint node:true */

"use strict";

var sequence = function (tasks, names, results, nest) {
	var i, name, node, e, j;
	nest = nest || [];
	for (i = 0; i < names.length; i++) {
		name = names[i];
		// de-dup results
		if (results.indexOf(name) === -1) {
			node = tasks[name];
			if (!node) {
				e = new Error('task "'+name+'" is not defined');
				e.missingTask = name;
				e.taskList = [];
				for (j in tasks) {
					if (tasks.hasOwnProperty(j)) {
						e.taskList.push(tasks[j].name);
					}
				}
				throw e;
			}
			if (nest.indexOf(name) > -1) {
				nest.push(name);
				e = new Error('Recursive dependencies detected: '+nest.join(' -> '));
				e.recursiveTasks = nest;
				e.taskList = [];
				for (j in tasks) {
					if (tasks.hasOwnProperty(j)) {
						e.taskList.push(tasks[j].name);
					}
				}
				throw e;
			}
			if (node.dep.length) {
				nest.push(name);
				sequence(tasks, node.dep, results, nest); // recurse
				nest.pop(name);
			}
			results.push(name);
		}
	}
};

module.exports = sequence;
