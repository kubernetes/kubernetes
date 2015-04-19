var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('scan', function (t) {
    t.plan(4);
    
    var em = new EventEmitter;
    binary(em)
        .word8('a')
        .scan('l1', new Buffer('\r\n'))
        .scan('l2', '\r\n')
        .word8('z')
        .tap(function (vars) {
            t.same(vars.a, 99);
            t.same(vars.l1.toString(), 'foo bar');
            t.same(vars.l2.toString(), 'baz');
            t.same(vars.z, 42);
        })
    ;
    
    setTimeout(function () {
        em.emit('data', new Buffer([99,0x66,0x6f,0x6f,0x20]));
    }, 20);
    
    setTimeout(function () {
        em.emit('data', new Buffer('bar\r'));
    }, 40);
    
    setTimeout(function () {
        em.emit('data', new Buffer('\nbaz\r\n*'));
    }, 60);
});
