var binary = require('../');
var test = require('tap').test;

test('get buffer', function (t) {
    t.plan(4);
    
    var buf = new Buffer([ 4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ]);
    binary.parse(buf)
        .word8('a')
        .buffer('b', 7)
        .word16lu('c')
        .tap(function (vars) {
            t.equal(vars.a, 4);
            t.equal(
                vars.b.toString(), 
                new Buffer([ 2, 3, 4, 5, 6, 7, 8 ]).toString()
            );
            t.equal(vars.c, 2569);
        })
        .buffer('d', 'a')
        .tap(function (vars) {
            t.equal(
                vars.d.toString(),
                new Buffer([ 11, 12, 13, 14 ]).toString()
            );
        })
    ;
});
