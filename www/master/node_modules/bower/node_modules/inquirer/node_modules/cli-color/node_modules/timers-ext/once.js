'use strict';

var callable     = require('es5-ext/object/valid-callable')
  , nextTick     = require('next-tick')
  , validTimeout = require('./valid-timeout');

module.exports = function (fn/*, timeout*/) {
	var scheduled, run, context, args, delay, index, timeout = arguments[1];
	callable(fn);
	if (timeout === undefined) {
		delay = nextTick;
	} else {
		timeout = validTimeout(timeout);
		delay = setTimeout;
	}
	run = function () {
		scheduled = false;
		index = null;
		fn.apply(context, args);
		context = null;
		args = null;
	};
	return function () {
		if (scheduled) {
			if (index == null) return;
			clearTimeout(index);
		}
		scheduled = true;
		context = this;
		args = arguments;
		index = delay(run, timeout);
	};
};
