var assert = require('assert');
var Traverse = require('traverse');
var EventEmitter = require('events').EventEmitter;

exports['check instanceof on node elems'] = function () {
    
    var counts = { emitter : 0 };
    
    Traverse([ new EventEmitter, 3, 4, { ev : new EventEmitter }])
        .forEach(function (node) {
            if (node instanceof EventEmitter) counts.emitter ++;
        })
    ;
    
    assert.equal(counts.emitter, 2);
};

