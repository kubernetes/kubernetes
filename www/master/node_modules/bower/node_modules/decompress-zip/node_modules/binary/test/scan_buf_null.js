var binary = require('../');
var test = require('tap').test;

test('scan buf null', function (t) {
    t.plan(3);
    var vars = binary(new Buffer('\x63foo bar baz'))
        .word8('a')
        .scan('b', '\r\n')
        .word8('c')
        .vars
    ;
    
    t.same(vars.a, 99);
    t.same(vars.b.toString(), 'foo bar baz');
    t.strictEqual(vars.c, null);
});
