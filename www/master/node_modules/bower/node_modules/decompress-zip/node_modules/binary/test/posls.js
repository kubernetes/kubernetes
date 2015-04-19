var binary = require('../');
var test = require('tap').test;

test('posls', function (t) {
    t.plan(4);
    
    // note: can't store 12667700813876161 exactly in an ieee float
    var buf = new Buffer([
        30, // a == -30
        37, 9, // b == -2341
        20, 10, 12, 0, // c == -789012
        193, 203, 33, 239, 52, 1, 45, 0, // d == 12667700813876161
    ]);
    
    binary.parse(buf)
        .word8ls('a')
        .word16ls('b')
        .word32ls('c')
        .word64ls('d')
        .tap(function (vars) {
            t.same(vars.a, 30);
            t.same(vars.b, 2341);
            t.same(vars.c, 789012);
            t.ok(
                Math.abs(vars.d - 12667700813876161) < 1000
            );
        })
    ;
});
