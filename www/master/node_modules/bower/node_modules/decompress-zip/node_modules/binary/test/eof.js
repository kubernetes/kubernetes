var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('eof', function (t) {
    t.plan(4);
    
    var stream = new EventEmitter;
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
    
    var bufs = [
        new Buffer([ 6, 1, 6, 1, 6, 9, 0, 0, 0, 97 ]),
        new Buffer([ 98, 99 ]),
        new Buffer([ 100, 101, 102 ]),
    ];
    
    bufs.forEach(function (buf) {
        stream.emit('data', buf);
    });
    
    stream.emit('end');
});
