var binary = require('../');
var test = require('tap').test;

test('posbs', function (t) {
    t.plan(4);
    // note: can't store 12667700813876161 exactly in an ieee float
    
    var buf = new Buffer([
        30, // a == -30
        9, 37, // b == -2341
        0, 12, 10, 20, // c == -789012
        0, 45, 1, 52, 239, 33, 203, 193, // d == 12667700813876161
    ]);
    
    binary.parse(buf)
        .word8bs('a')
        .word16bs('b')
        .word32bs('c')
        .word64bs('d')
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
