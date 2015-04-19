var binary = require('../');
var test = require('tap').test;

test('negbs', function (t) {
    t.plan(4);
    // note: can't store -12667700813876161 exactly in an ieee float
    
    var buf = new Buffer([
        226, // a == -30
        246, 219, // b == -2341
        255, 243, 245, 236, // c == -789012
        255, 210, 254, 203, 16, 222, 52, 63, // d == -12667700813876161
    ]);
    
    binary.parse(buf)
        .word8bs('a')
        .word16bs('b')
        .word32bs('c')
        .word64bs('d')
        .tap(function (vars) {
            t.same(vars.a, -30);
            t.same(vars.b, -2341);
            t.same(vars.c, -789012);
            t.ok(
                Math.abs(vars.d - -12667700813876161) < 1500
            );
        })
    ;
});
