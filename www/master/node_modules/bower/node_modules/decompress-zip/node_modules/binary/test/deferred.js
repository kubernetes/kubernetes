var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('deferred', function (t) {
    t.plan(1);
    
    var em = new EventEmitter;
    binary.stream(em)
        .word8('a')
        .word16be('bc')
        .tap(function (vars) {
            t.same(vars, { a : 97, bc : 25187 });
        })
    ;
    
    setTimeout(function () {
        em.emit('data', new Buffer([ 97, 98, 99 ]));
    }, 10);
});
