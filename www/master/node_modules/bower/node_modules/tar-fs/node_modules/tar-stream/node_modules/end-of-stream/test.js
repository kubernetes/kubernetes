var assert = require('assert');
var eos = require('./index');

var expected = 8;
var fs = require('fs');
var cp = require('child_process');
var net = require('net');

var ws = fs.createWriteStream('/dev/null');
eos(ws, function(err) {
	expected--;
	assert(!!err);
	if (!expected) process.exit(0);
});
ws.close();

var rs = fs.createReadStream('/dev/random');
eos(rs, function(err) {
	expected--;
	assert(!!err);
	if (!expected) process.exit(0);
});
rs.close();

var rs = fs.createReadStream(__filename);
eos(rs, function(err) {
	expected--;
	assert(!err);
	if (!expected) process.exit(0);
});
rs.pipe(fs.createWriteStream('/dev/null'));

var rs = fs.createReadStream(__filename);
eos(rs, function(err) {
	throw new Error('no go')
})();
rs.pipe(fs.createWriteStream('/dev/null'));

var exec = cp.exec('echo hello world');
eos(exec, function(err) {
	expected--;
	assert(!err);
	if (!expected) process.exit(0);
});

var spawn = cp.spawn('echo', ['hello world']);
eos(spawn, function(err) {
	expected--;
	assert(!err);
	if (!expected) process.exit(0);
});

var socket = net.connect(50000);
eos(socket, function(err) {
	expected--;
	assert(!!err);
	if (!expected) process.exit(0);
});

var server = net.createServer(function(socket) {
	eos(socket, function() {
		expected--;
		if (!expected) process.exit(0);
	});
	socket.destroy();
}).listen(30000, function() {
	var socket = net.connect(30000);
	eos(socket, function() {
		expected--;
		if (!expected) process.exit(0);
	});
});

setTimeout(function() {
	assert(expected === 0);
	process.exit(0);
}, 1000);
