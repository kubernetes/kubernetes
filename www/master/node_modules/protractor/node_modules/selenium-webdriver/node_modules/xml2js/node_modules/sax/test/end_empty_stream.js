var assert = require('assert');
var saxStream = require('../lib/sax').createStream();
assert.doesNotThrow(function() {
    saxStream.end();
});
