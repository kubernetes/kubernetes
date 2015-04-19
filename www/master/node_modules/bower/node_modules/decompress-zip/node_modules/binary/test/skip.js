var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;
var seq = require('seq');

test('skip', function (t) {
    t.plan(7);
    var em = new EventEmitter;
    var state = 0;
    
    binary(em)
        .word16lu('a')
        .tap(function () { state = 1 })
        .skip(7)
        .tap(function () { state = 2 })
        .word8('b')
        .tap(function () { state = 3 })
        .tap(function (vars) {
            t.same(state, 3);
            t.same(vars, {
                a : 2569,
                b : 8,
            });
        })
    ;
    
    seq()
        .seq(setTimeout, seq, 20)
        .seq(function () {
            t.same(state, 0);
            em.emit('data', new Buffer([ 9 ]));
            this(null);
        })
        .seq(setTimeout, seq, 5)
        .seq(function () {
            t.same(state, 0);
            em.emit('data', new Buffer([ 10, 1, 2 ]));
            this(null);
        })
        .seq(setTimeout, seq, 30)
        .seq(function () {
            t.same(state, 1);
            em.emit('data', new Buffer([ 3, 4, 5 ]));
            this(null);
        })
        .seq(setTimeout, seq, 15)
        .seq(function () {
            t.same(state, 1);
            em.emit('data', new Buffer([ 6, 7 ]));
            this(null);
        })
        .seq(function () {
            t.same(state, 2);
            em.emit('data', new Buffer([ 8 ]));
            this(null);
        })
    ;
});
