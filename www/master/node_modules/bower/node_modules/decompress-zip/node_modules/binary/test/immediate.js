var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('immediate', function (t) {
    t.plan(1);
    
    var em = new EventEmitter;
    binary.stream(em, 'moo')
        .word8('a')
        .word16be('bc')
        .tap(function (vars) {
            t.same(vars, { a : 97, bc : 25187 });
        })
    ;
    
    em.emit('moo', new Buffer([ 97, 98, 99 ]));
});
