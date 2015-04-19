'use strict';

var startsWith = require('es5-ext/string/#/starts-with')
  , spawn      = require('child_process').spawn
  , resolve    = require('path').resolve
  , pg = resolve(__dirname, '__playground');

module.exports = {
	"": function (a, d) {
		var t = spawn('node', [resolve(pg, 'throbber.js')])
		  , out = [], err = '';

		t.stdout.on('data', function (data) {
			out.push(data);
		});
		t.stderr.on('data', function (data) {
			err += data;
		});
		t.on('exit', function () {
			a.ok(out.length > 4, "Interval");
			a(startsWith.call(out.join(""), "START-\b\\\b|\b/\b-\b"), true, "Output");
			a(err, "", "No stderr output");
			d();
		});
	},
	Formatted: function (a, d) {
		var t = spawn('node', [resolve(pg, 'throbber.formatted.js')])
		  , out = [], err = '';

		t.stdout.on('data', function (data) {
			out.push(data);
		});
		t.stderr.on('data', function (data) {
			err += data;
		});
		t.on('exit', function () {
			a.ok(out.length > 4, "Interval");
			a(startsWith.call(out.join(""), "START\x1b[31m-\x1b[39m\x1b[31m\b\\\x1b" +
				"[39m\x1b[31m\b|\x1b[39m\x1b[31m\b/\x1b[39m\x1b[31m\b-\x1b[39m"),
				true, "Output");
			a(err, "", "No stderr output");
			d();
		});
	}
};
