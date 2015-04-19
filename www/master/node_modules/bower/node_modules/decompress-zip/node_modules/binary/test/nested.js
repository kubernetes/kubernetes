var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('nested', function (t) {
    t.plan(3);
    var insideDone = false;
    
    var em = new EventEmitter;
    binary.stream(em)
        .word16be('ab')
        .tap(function () {
            this
                .word8('c')
                .word8('d')
                .tap(function () {
                    insideDone = true;
                })
            ;
        })
        .tap(function (vars) {
            t.ok(insideDone);
            t.same(vars.c, 'c'.charCodeAt(0));
            t.same(vars.d, 'd'.charCodeAt(0));
            
        })
    ;
    
    var strs = [ 'abc', 'def', 'hi', 'jkl' ];
    var iv = setInterval(function () {
        var s = strs.shift();
        if (s) em.emit('data', new Buffer(s));
        else clearInterval(iv);
    }, 50);
});
