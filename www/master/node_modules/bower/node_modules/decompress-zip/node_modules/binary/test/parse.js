var binary = require('../');
var test = require('tap').test;

test('parse', function (t) {
    t.plan(6);
    var res = binary.parse(new Buffer([ 97, 98, 99, 99, 99, 99, 1, 2, 3 ]))
        .word8('a')
        .word16be('bc')
        .skip(3)
        .buffer('def', 3)
        .tap(function (vars) {
            t.equal(vars.a, 97);
            t.equal(vars.bc, 25187);
            t.same(
                [].slice.call(vars.def),
                [].slice.call(new Buffer([ 1, 2, 3]))
            );
        })
        .vars
    ;
    t.equal(res.a, 97);
    t.equal(res.bc, 25187);
    t.same(
        [].slice.call(res.def),
        [].slice.call(new Buffer([ 1, 2, 3 ]))
    );
});

test('loop', function (t) {
    t.plan(2);
    var res = binary.parse(new Buffer([ 97, 98, 99, 4, 5, 2, -3, 9 ]))
        .word8('a')
        .word16be('bc')
        .loop(function (end) {
            var x = this.word8s('x').vars.x;
            if (x < 0) end();
        })
        .tap(function (vars) {
            t.same(vars, {
                a : 97,
                bc : 25187,
                x : -3,
            });
        })
        .word8('y')
        .vars
    ;
    t.same(res, {
        a : 97,
        bc : 25187,
        x : -3,
        y : 9,
    });
});
