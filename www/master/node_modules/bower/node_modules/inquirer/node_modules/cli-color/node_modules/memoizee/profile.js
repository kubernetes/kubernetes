// Gathers statistical data, and provides them in convinient form

'use strict';

var partial  = require('es5-ext/function/#/partial')
  , forEach  = require('es5-ext/object/for-each')
  , pad      = require('es5-ext/string/#/pad')
  , d        = require('d')
  , memoize  = require('./plain')

  , max = Math.max
  , stats = exports.statistics = {};

Object.defineProperty(memoize, '__profiler__', d(function (conf) {
	var id, stack, data;
	stack = (new Error()).stack;
	if (!stack || !stack.split('\n').slice(3).some(function (line) {
			if ((line.indexOf('/memoizee/') === -1) &&
					(line.indexOf(' (native)') === -1)) {
				id  = line.replace(/\n/g, "\\n").trim();
				return true;
			}
		})) {
		id = 'unknown';
	}

	if (!stats[id]) stats[id] = { initial: 0, cached: 0 };
	data = stats[id];

	conf.on('set', function () { ++data.initial; });
	conf.on('get', function () { ++data.cached; });
}));

exports.log = function () {
	var initial, cached, ordered, ipad, cpad, ppad, toPrc, log;

	initial = cached = 0;
	ordered = [];

	toPrc = function (initial, cached) {
		if (!initial && !cached) {
			return '0.00';
		}
		return ((cached / (initial + cached)) * 100).toFixed(2);
	};

	log = "------------------------------------------------------------\n";
	log += "Memoize statistics:\n\n";

	forEach(stats, function (data, name) {
		initial += data.initial;
		cached += data.cached;
		ordered.push([name, data]);
	}, null, function (a, b) {
		return (this[b].initial + this[b].cached) -
			(this[a].initial + this[a].cached);
	});

	ipad = partial.call(pad, " ",
		max(String(initial).length, "Init".length));
	cpad = partial.call(pad, " ", max(String(cached).length, "Cache".length));
	ppad = partial.call(pad, " ", "%Cache".length);
	log += ipad.call("Init") + "  " +
		cpad.call("Cache") + "  " +
		ppad.call("%Cache") + "  Source location\n";
	log += ipad.call(initial) + "  " +
		cpad.call(cached) + "  " +
		ppad.call(toPrc(initial, cached)) + "  (all)\n";
	ordered.forEach(function (data) {
		var name = data[0];
		data = data[1];
		log += ipad.call(data.initial) + "  " +
			cpad.call(data.cached) + "  " +
			ppad.call(toPrc(data.initial, data.cached)) + "  " + name + "\n";
	});
	log += "------------------------------------------------------------\n";
	return log;
};
