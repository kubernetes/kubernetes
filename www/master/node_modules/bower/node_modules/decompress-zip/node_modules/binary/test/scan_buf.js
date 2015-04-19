var binary = require('../');
var test = require('tap').test;

test('scan buf', function (t) {
    t.plan(4);
    
    var vars = binary(new Buffer('\x63foo bar\r\nbaz\r\n*'))
        .word8('a')
        .scan('l1', new Buffer('\r\n'))
        .scan('l2', '\r\n')
        .word8('z')
        .vars
    ;
    t.same(vars.a, 99);
    t.same(vars.z, 42);
    t.same(vars.l1.toString(), 'foo bar');
    t.same(vars.l2.toString(), 'baz');
});
