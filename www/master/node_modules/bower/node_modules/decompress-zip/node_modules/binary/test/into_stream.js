var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('into stream', function (t) {
    t.plan(3);
    
    var digits = [ 1, 2, 3, 4, 5, 6 ];
    var stream = new EventEmitter;
    var iv = setInterval(function () {
        var d = digits.shift();
        if (d) stream.emit('data', new Buffer([ d ]))
        else clearInterval(iv)
    }, 20);
    
    binary.stream(stream)
        .into('moo', function () {
            this
                .word8('x')
                .word8('y')
                .word8('z')
            ;
        })
        .tap(function (vars) {
            t.same(vars, { moo : { x : 1, y : 2, z : 3 } });
        })
        .word8('w')
        .tap(function (vars) {
            t.same(vars, {
                moo : { x : 1, y : 2, z : 3 },
                w : 4,
            });
        })
        .word8('x')
        .tap(function (vars) {
            t.same(vars, {
                moo : { x : 1, y : 2, z : 3 },
                w : 4,
                x : 5,
            });
        })
    ;
});
