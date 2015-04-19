var binary = require('../');
var test = require('tap').test;
var EventEmitter = require('events').EventEmitter;

test('loop', function (t) {
    t.plan(3 * 2 + 1);
    
    var em = new EventEmitter;
    
    binary.stream(em)
        .loop(function (end, vars) {
            t.strictEqual(vars, this.vars);
            this
                .word16lu('a')
                .word8u('b')
                .word8s('c')
                .tap(function (vars_) {
                    t.strictEqual(vars, vars_);
                    if (vars.c < 0) end();
                })
            ;
        })
        .tap(function (vars) {
            t.same(vars, { a : 1337, b : 55, c : -5 });
        })
    ;
    
    setTimeout(function () {
        em.emit('data', new Buffer([ 2, 10, 88 ]));
    }, 10);
    setTimeout(function () {
        em.emit('data', new Buffer([ 100, 3, 6, 242, 30 ]));
    }, 20);
    setTimeout(function () {
        em.emit('data', new Buffer([ 60, 60, 199, 44 ]));
    }, 30);
    
    setTimeout(function () {
        em.emit('data', new Buffer([ 57, 5 ]));
    }, 80);
    setTimeout(function () {
        em.emit('data', new Buffer([ 55, 251 ]));
    }, 90);
});
