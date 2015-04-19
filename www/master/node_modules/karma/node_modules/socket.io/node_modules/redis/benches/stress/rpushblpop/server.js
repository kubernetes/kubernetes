'use strict';

var freemem = require('os').freemem;
var codec = require('../codec');

var id = Math.random();
var recv = 0;

var cmd = require('redis').createClient();
var sub = require('redis').createClient()
	.on('ready', function() {
		this.emit('timeline');
	})
	.on('timeline', function() {
		var self = this;
		this.blpop('timeline', 0, function(err, result) {
			var message = result[1];
			if (message) {
				message = codec.decode(message);
				++recv;
			}
			self.emit('timeline');
		});
	});

setInterval(function() {
	cmd.llen('timeline', function(err, result) {
		console.error('id', id, 'received', recv, 'free', freemem(), 'llen', result);
	});
}, 2000);
