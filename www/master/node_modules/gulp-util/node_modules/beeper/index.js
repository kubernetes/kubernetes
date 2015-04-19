'use strict';

var BEEP_DELAY = 500;

if (!process.stdout.isTTY ||
	process.argv.indexOf('--no-beep') !== -1 ||
	process.argv.indexOf('--beep=false') !== -1) {
	module.exports = function () {};
	return;
}

function beep() {
	process.stdout.write('\u0007');
}

function melodicalBeep(val) {
	if (val.length === 0) {
		return;
	}

	setTimeout(function () {
		if (val.shift() === '*') {
			beep();
		}

		melodicalBeep(val);
	}, BEEP_DELAY);
}

module.exports = function (val) {
	if (val == null) {
		beep();
	} else if (typeof val === 'number') {
		beep();
		val--;

		while (val--) {
			setTimeout(beep, BEEP_DELAY * val);
		}
	} else if (typeof val === 'string') {
		melodicalBeep(val.split(''));
	} else {
		throw new TypeError('Not an accepted type');
	}
};
