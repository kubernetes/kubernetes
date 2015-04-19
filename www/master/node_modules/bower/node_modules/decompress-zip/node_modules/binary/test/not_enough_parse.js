var binary = require('../');
var test = require('tap').test;

test('not enough parse', function (t) {
    t.plan(4);
    
    var vars = binary(new Buffer([1,2]))
        .word8('a')
        .word8('b')
        .word8('c')
        .word8('d')
        .vars
    ;
    
    t.same(vars.a, 1);
    t.same(vars.b, 2);
    t.strictEqual(vars.c, null);
    t.strictEqual(vars.d, null);
});
