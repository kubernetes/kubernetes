var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('interval', function (t) {
    t.plan(1);
    
    var em = new EventEmitter;
    var i = 0;
    var iv = setInterval(function () {
        var buf = new Buffer(1000);
        buf[0] = 0xff;
        if (++i >= 1000) {
            clearInterval(iv);
            buf[0] = 0;
        }
        em.emit('data', buf);
    }, 1);
    
    var loops = 0;
    binary(em)
        .loop(function (end) {
            this
            .word8('x')
            .word8('y')
            .word32be('z')
            .word32le('w')
            .buffer('buf', 1000 - 10)
            .tap(function (vars) {
                loops ++;
                if (vars.x == 0) end();
            })
        })
        .tap(function () {
            t.same(loops, 1000);
        })
    ;
});
