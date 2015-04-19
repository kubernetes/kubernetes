var binary = require('../');
var test = require('tap').test;

test('intoBuffer', function (t) {
    t.plan(3);
    var buf = new Buffer([ 1, 2, 3, 4, 5, 6 ])
    
    binary.parse(buf)
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
