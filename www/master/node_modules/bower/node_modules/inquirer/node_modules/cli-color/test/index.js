'use strict';

module.exports = function (t, a) {
	var x, y;

	a(t('test'), 'test', "Plain");
	a(t('test', 'foo', 3, { toString: function () { return 'bar'; } }),
		'test foo 3 bar', "Plain: Many args");

	a(t.red('foo'), '\x1b[31mfoo\x1b[39m', "Foreground");
	a(t.red('foo', 'bar', 3), '\x1b[31mfoo bar 3\x1b[39m',
		"Foreground: Many args");
	a(t.red.yellow('foo', 'bar', 3), '\x1b[33mfoo bar 3\x1b[39m',
		"Foreground: Overriden");
	a(t.bgRed('foo', 'bar'), '\x1b[41mfoo bar\x1b[49m', "Background");
	a(t.bgRed.bgYellow('foo', 'bar', 3), '\x1b[43mfoo bar 3\x1b[49m',
		"Background: Overriden");

	a(t.blue.bgYellow('foo', 'bar'), '\x1b[43m\x1b[34mfoo bar\x1b[39m\x1b[49m',
		"Foreground & Background");
	a(t.blue.bgYellow.red.bgMagenta('foo', 'bar'),
		'\x1b[45m\x1b[31mfoo bar\x1b[39m\x1b[49m',
		"Foreground & Background: Overriden");

	a(t.bold('foo', 'bar'), '\x1b[1mfoo bar\x1b[22m', "Format");
	a(t.blink('foobar'), '\x1b[5mfoobar\x1b[25m', "Format: blink");
	a(t.bold.blue('foo', 'bar', 3), '\x1b[1m\x1b[34mfoo bar 3\x1b[39m\x1b[22m',
		"Foreground & Format");

	a(t.redBright('foo', 'bar'), '\x1b[91mfoo bar\x1b[39m', "Bright");
	a(t.bgRedBright('foo', 3), '\x1b[101mfoo 3\x1b[49m', "Bright background");

	a(t.blueBright.bgYellowBright.red.bgMagenta('foo', 'bar'),
		'\x1b[45m\x1b[31mfoo bar\x1b[39m\x1b[49m',
		"Foreground & Background: Bright: Overriden");

	x = t.red;
	y = x.bold;

	a(x('foo', 'red') + ' ' + y('foo', 'boldred'),
		'\x1b[31mfoo red\x1b[39m \x1b[1m\x1b[31mfoo boldred\x1b[39m\x1b[22m',
		"Detached extension");

	if (t.xtermSupported) {
		a(t.xterm(12).bgXterm(67)('foo', 'xterm'),
			'\x1b[48;5;67m\x1b[38;5;12mfoo xterm\x1b[39m\x1b[49m', "Xterm");

		a(t.redBright.bgBlueBright.xterm(12).bgXterm(67)('foo', 'xterm'),
			'\x1b[48;5;67m\x1b[38;5;12mfoo xterm\x1b[39m\x1b[49m',
			"Xterm: Override & Bright");
		a(t.xterm(12).bgXterm(67).redBright.bgMagentaBright('foo', 'xterm'),
			'\x1b[105m\x1b[91mfoo xterm\x1b[39m\x1b[49m',
			"Xterm: Override & Bright #2");
	} else {
		a(t.xterm(12).bgXterm(67)('foo', 'xterm'),
			'\x1b[100m\x1b[94mfoo xterm\x1b[39m\x1b[49m', "Xterm");
	}

	a(typeof t.width, 'number', "Width");
	a(typeof t.height, 'number', "Height");

	a(/\n*\x1bc/.test(t.reset), true, "Reset");

	a(t.up(), '', "Up: No argument");
	a(t.up({}), '', "Up: Not a number");
	a(t.up(-34), '', "Up: Negative");
	a(t.up(34), '\x1b[34A', "Up: Positive");

	a(t.down(), '', "Down: No argument");
	a(t.down({}), '', "Down: Not a number");
	a(t.down(-34), '', "Down: Negative");
	a(t.down(34), '\x1b[34B', "Down: Positive");

	a(t.right(), '', "Right: No argument");
	a(t.right({}), '', "Right: Not a number");
	a(t.right(-34), '', "Right: Negative");
	a(t.right(34), '\x1b[34C', "Right: Positive");

	a(t.left(), '', "Left: No argument");
	a(t.left({}), '', "Left: Not a number");
	a(t.left(-34), '', "Left: Negative");
	a(t.left(34), '\x1b[34D', "Left: Positive");

	a(t.move(), '', "Move: No arguments");
	a(t.move({}, {}), '', "Move: Bad arguments");
	a(t.move({}, 12), '\x1b[12B', "Move: One direction");
	a(t.move(0, -12), '\x1b[12A', "Move: One negative direction");
	a(t.move(-42, -2), '\x1b[42D\x1b[2A', "Move: two negatives");
	a(t.move(2, 35), '\x1b[2C\x1b[35B', "Move: two positives");

	a(t.moveTo(), '\x1b[1;1H', "MoveTo: No arguments");
	a(t.moveTo({}, {}), '\x1b[1;1H', "MoveTo: Bad arguments");
	a(t.moveTo({}, 12), '\x1b[13;1H', "MoveTo: One direction");
	a(t.moveTo(2, -12), '\x1b[1;3H', "MoveTo: One negative direction");
	a(t.moveTo(-42, -2), '\x1b[1;1H', "MoveTo: two negatives");
	a(t.moveTo(2, 35), '\x1b[36;3H', "MoveTo: two positives");

	a(t.bol(), '\x1b[0E', "Bol: No argument");
	a(t.bol({}), '\x1b[0E', "Bol: Not a number");
	a(t.bol(-34), '\x1b[34F', "Bol: Negative");
	a(t.bol(34), '\x1b[34E', "Bol: Positive");

	a(t.bol({}, true), '\x1b[0E\x1bK', "Bol: Erase: Not a number");
	a(t.bol(-2, true), '\x1b[0E\x1bK\x1b[1F\x1b[K\x1b[1F\x1b[K',
		"Bol: Erase: Negative");
	a(t.bol(2, true), '\x1b[1E\x1b[K\x1b[1E\x1b[K',
		"Bol: Erase: Positive");

	a(t.beep, '\x07', "Beep");
};
