var binary = require('../');
var test = require('tap').test;

test('bu', function (t) {
    t.plan(8);
    
    // note: can't store -12667700813876161 exactly in an ieee float
    
    var buf = new Buffer([
        44, // a == 44
        2, 43, // b == 555
        164, 213, 37, 37, // c == 2765432101
        29, 81, 180, 20, 155, 115, 203, 193, // d == 2112667700813876161
    ]);
    
    binary.parse(buf)
        .word8bu('a')
        .word16bu('b')
        .word32bu('c')
        .word64bu('d')
        .tap(function (vars) {
            t.same(vars.a, 44);
            t.same(vars.b, 555);
            t.same(vars.c, 2765432101);
            t.ok(
                Math.abs(vars.d - 2112667700813876161) < 1500
            );
        })
    ;
    
    // also check aliases here:
    binary.parse(buf)
        .word8be('a')
        .word16be('b')
        .word32be('c')
        .word64be('d')
        .tap(function (vars) {
            t.same(vars.a, 44);
            t.same(vars.b, 555);
            t.same(vars.c, 2765432101);
            t.ok(
                Math.abs(vars.d - 2112667700813876161) < 1500
            );
        })
    ;
});
