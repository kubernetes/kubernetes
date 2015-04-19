var binary = require('../');
var test = require('tap').test;

test('dots', function (t) {
    t.plan(1);
    
    binary.parse(new Buffer([ 97, 98, 99, 100, 101, 102 ]))
        .word8('a')
        .word16be('b.x')
        .word16be('b.y')
        .word8('b.z')
        .tap(function (vars) {
            t.same(vars, {
                a : 97,
                b : {
                    x : 256 * 98 + 99,
                    y : 256 * 100 + 101,
                    z : 102
                },
            });
        })
    ;
});
