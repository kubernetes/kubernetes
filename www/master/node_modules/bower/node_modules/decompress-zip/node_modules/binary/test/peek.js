var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('peek', function (t) {
    t.plan(4);
    var bufs = [
        new Buffer([ 6, 1, 6, 1, 6, 9, 0, 0, 0, 97 ]),
        new Buffer([ 98, 99 ]),
        new Buffer([ 100, 101, 102 ]),
    ];
    
    var stream = new EventEmitter;
    var iv = setInterval(function () {
        var buf = bufs.shift();
        if (buf) stream.emit('data', buf)
        else clearInterval(iv)
    }, 20);
    
    binary.stream(stream)
        .buffer('sixone', 5)
        .peek(function () {
            this.word32le('len');
        })
        .buffer('buf', 'len')
        .word8('x')
        .tap(function (vars) {
            t.same(
                [].slice.call(vars.sixone),
                [].slice.call(new Buffer([ 6, 1, 6, 1, 6 ]))
            );
            t.same(vars.buf.length, vars.len);
            t.same(
                [].slice.call(vars.buf),
                [ 9, 0, 0, 0, 97, 98, 99, 100, 101 ]
            );
            t.same(vars.x, 102);
        })
    ;
});
