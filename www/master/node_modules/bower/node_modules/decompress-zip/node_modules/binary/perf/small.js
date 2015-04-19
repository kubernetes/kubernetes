var Seq = require('seq');
var Hash = require('hashish');

var Bin = require('binary');
var Buf = require('bufferlist/binary');
var BufferList = require('bufferlist');
var EventEmitter = require('events').EventEmitter;

function binary (buf, cb) {
    Bin(buf)
        .word32le('x')
        .word16be('y')
        .word16be('z')
        .word32le('w')
        .tap(cb)
    ;
};

function stream (buf, cb) {
    var em = new EventEmitter;
    Bin(em)
        .word32le('x')
        .word16be('y')
        .word16be('z')
        .word32le('w')
        .tap(cb)
    ;
    em.emit('data', buf);
};

function parse (buf, cb) {
    cb(Bin.parse(buf)
        .word32le('x')
        .word16be('y')
        .word16be('z')
        .word32le('w')
        .vars
    );
};

function bufferlist (buf, cb) {
    var blist = new BufferList;
    blist.push(buf);
    Buf(blist)
        .getWord32le('x')
        .getWord16be('y')
        .getWord16be('z')
        .getWord32le('w')
        .tap(cb)
        .end()
    ;
};


var buffers = [];
for (var i = 0; i < 200; i++) {
    buffers.push(new Buffer(12));
}

console.log('small');
Seq(binary, stream, parse, bufferlist)
    .seqEach(function (f) {
        var t = this;
        var t0 = Date.now();
        Seq()
            .extend(buffers)
            .seqEach(function (buf) {
                f(buf, this.bind(this, null));
            })
            .seq(function () {
                var tf = Date.now();
                console.log('    ' + f.name + ': ' + (tf - t0));
                t(null);
            })
        ;
    })
    .seq(function () {
        this(null);
    })
;
