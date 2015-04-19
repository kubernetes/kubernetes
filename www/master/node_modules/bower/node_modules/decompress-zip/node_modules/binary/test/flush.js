var binary = require('../');
var test = require('tap').test;

test('flush', function (t) {
    t.plan(1);
    
    binary.parse(new Buffer([ 97, 98, 99, 100, 101, 102 ]))
        .word8('a')
        .word16be('b')
        .word16be('c')
        .flush()
        .word8('d')
        .tap(function (vars) {
            t.same(vars, { d : 102 });
        })
    ;
});
