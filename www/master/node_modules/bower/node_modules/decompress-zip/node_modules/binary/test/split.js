var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('split', function (t) {
    t.plan(1);
    
    var em = new EventEmitter;
    binary.stream(em)
        .word8('a')
        .word16be('bc')
        .word32ls('x')
        .word32bs('y')
        .tap(function (vars) {
            t.same(vars, {
                a : 97,
                bc : 25187,
                x : 621609828,
                y : 621609828,
            });
        })
    ;
    
    em.emit('data', new Buffer([ 97, 98 ]));
    setTimeout(function () {
        em.emit('data', new Buffer([ 99, 100 ]));
    }, 25);
    setTimeout(function () {
        em.emit('data', new Buffer([ 3, 13, 37, 37 ]));
    }, 30);
    setTimeout(function () {
        em.emit('data', new Buffer([ 13, 3, 100 ]));
    }, 40);
});
