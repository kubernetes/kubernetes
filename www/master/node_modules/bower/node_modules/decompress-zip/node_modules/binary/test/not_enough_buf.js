var binary = require('../');
var test = require('tap').test;

test('not enough buf', function (t) {
    t.plan(3);
    
    var vars = binary(new Buffer([1,2,3,4]))
        .word8('a')
        .buffer('b', 10)
        .word8('c')
        .vars
    ;
    
    t.same(vars.a, 1);
    t.equal(vars.b.toString(), new Buffer([2,3,4]).toString());
    t.strictEqual(vars.c, null);
});
