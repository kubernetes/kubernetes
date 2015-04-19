'use strict';
var request = require('request');
var async = require('async');
var assign = require('object-assign');
var Insight = require('./');

// Messaged on each debounced track()
// Gets the queue, merges is with the previous and tries to upload everything
// If it fails, it will save everything again
process.on('message', function (msg) {
	var insight = new Insight(msg);
	var config = insight.config;
	var q = config.get('queue') || {};

	assign(q, msg.queue);
	config.del('queue');

	async.forEachSeries(Object.keys(q), function (el, cb) {
		var parts = el.split(' ');
		var id = parts[0];
		var path = parts[1];

		request(insight._getRequestObj(id, path), function (err, res, body) {
			if (err) {
				cb(err);
				return;
			}

			cb();
		});
	}, function (err) {
		if (err) {
			var q2 = config.get('queue') || {};
			assign(q2, q);
			config.set('queue', q2);
		}

		process.exit();
	});
});
