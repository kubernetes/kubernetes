var assert = require('assert');
var traverse = require('traverse');

exports.deepDates = function () {
    assert.ok(
        traverse.deepEqual(
            { d : new Date, x : [ 1, 2, 3 ] },
            { d : new Date, x : [ 1, 2, 3 ] }
        ),
        'dates should be equal'
    );
    
    var d0 = new Date;
    setTimeout(function () {
        assert.ok(
            !traverse.deepEqual(
                { d : d0, x : [ 1, 2, 3 ], },
                { d : new Date, x : [ 1, 2, 3 ] }
            ),
            'microseconds should count in date equality'
        );
    }, 5);
};

exports.deepCircular = function () {
    var a = [1];
    a.push(a); // a = [ 1, *a ]
    
    var b = [1];
    b.push(a); // b = [ 1, [ 1, *a ] ]
    
    assert.ok(
        !traverse.deepEqual(a, b),
        'circular ref mount points count towards equality'
    );
    
    var c = [1];
    c.push(c); // c = [ 1, *c ]
    assert.ok(
        traverse.deepEqual(a, c),
        'circular refs are structurally the same here'
    );
    
    var d = [1];
    d.push(a); // c = [ 1, [ 1, *d ] ]
    assert.ok(
        traverse.deepEqual(b, d),
        'non-root circular ref structural comparison'
    );
};

exports.deepInstances = function () {
    assert.ok(
        !traverse.deepEqual([ new Boolean(false) ], [ false ]),
        'boolean instances are not real booleans'
    );
    
    assert.ok(
        !traverse.deepEqual([ new String('x') ], [ 'x' ]),
        'string instances are not real strings'
    );
    
    assert.ok(
        !traverse.deepEqual([ new Number(4) ], [ 4 ]),
        'number instances are not real numbers'
    );
    
    assert.ok(
        traverse.deepEqual([ new RegExp('x') ], [ /x/ ]),
        'regexp instances are real regexps'
    );
    
    assert.ok(
        !traverse.deepEqual([ new RegExp(/./) ], [ /../ ]),
        'these regexps aren\'t the same'
    );
    
    assert.ok(
        !traverse.deepEqual(
            [ function (x) { return x * 2 } ],
            [ function (x) { return x * 2 } ]
        ),
        'functions with the same .toString() aren\'t necessarily the same'
    );
    
    var f = function (x) { return x * 2 };
    assert.ok(
        traverse.deepEqual([ f ], [ f ]),
        'these functions are actually equal'
    );
};

exports.deepEqual = function () {
    assert.ok(
        !traverse.deepEqual([ 1, 2, 3 ], { 0 : 1, 1 : 2, 2 : 3 }),
        'arrays are not objects'
    );
};

exports.falsy = function () {
    assert.ok(
        !traverse.deepEqual([ undefined ], [ null ]),
        'null is not undefined!'
    );
    
    assert.ok(
        !traverse.deepEqual([ null ], [ undefined ]),
        'undefined is not null!'
    );
    
    assert.ok(
        !traverse.deepEqual(
            { a : 1, b : 2, c : [ 3, undefined, 5 ] },
            { a : 1, b : 2, c : [ 3, null, 5 ] }
        ),
        'undefined is not null, however deeply!'
    );
    
    assert.ok(
        !traverse.deepEqual(
            { a : 1, b : 2, c : [ 3, undefined, 5 ] },
            { a : 1, b : 2, c : [ 3, null, 5 ] }
        ),
        'null is not undefined, however deeply!'
    );
    
    assert.ok(
        !traverse.deepEqual(
            { a : 1, b : 2, c : [ 3, undefined, 5 ] },
            { a : 1, b : 2, c : [ 3, null, 5 ] }
        ),
        'null is not undefined, however deeply!'
    );
};

exports.deletedArrayEqual = function () {
    var xs = [ 1, 2, 3, 4 ];
    delete xs[2];
    
    var ys = Object.create(Array.prototype);
    ys[0] = 1;
    ys[1] = 2;
    ys[3] = 4;
    
    assert.ok(
        traverse.deepEqual(xs, ys),
        'arrays with deleted elements are only equal to'
        + ' arrays with similarly deleted elements'
    );
    
    assert.ok(
        !traverse.deepEqual(xs, [ 1, 2, undefined, 4 ]),
        'deleted array elements cannot be undefined'
    );
    
    assert.ok(
        !traverse.deepEqual(xs, [ 1, 2, null, 4 ]),
        'deleted array elements cannot be null'
    );
};

exports.deletedObjectEqual = function () {
    var obj = { a : 1, b : 2, c : 3 };
    delete obj.c;
    
    assert.ok(
        traverse.deepEqual(obj, { a : 1, b : 2 }),
        'deleted object elements should not show up'
    );
    
    assert.ok(
        !traverse.deepEqual(obj, { a : 1, b : 2, c : undefined }),
        'deleted object elements are not undefined'
    );
    
    assert.ok(
        !traverse.deepEqual(obj, { a : 1, b : 2, c : null }),
        'deleted object elements are not null'
    );
};

exports.emptyKeyEqual = function () {
    assert.ok(!traverse.deepEqual(
        { a : 1 }, { a : 1, '' : 55 }
    ));
};

exports.deepArguments = function () {
    assert.ok(
        !traverse.deepEqual(
            [ 4, 5, 6 ],
            (function () { return arguments })(4, 5, 6)
        ),
        'arguments are not arrays'
    );
    
    assert.ok(
        traverse.deepEqual(
            (function () { return arguments })(4, 5, 6),
            (function () { return arguments })(4, 5, 6)
        ),
        'arguments should equal'
    );
};

exports.deepUn = function () {
    assert.ok(!traverse.deepEqual({ a : 1, b : 2 }, undefined));
    assert.ok(!traverse.deepEqual({ a : 1, b : 2 }, {}));
    assert.ok(!traverse.deepEqual(undefined, { a : 1, b : 2 }));
    assert.ok(!traverse.deepEqual({}, { a : 1, b : 2 }));
    assert.ok(traverse.deepEqual(undefined, undefined));
    assert.ok(traverse.deepEqual(null, null));
    assert.ok(!traverse.deepEqual(undefined, null));
};

exports.deepLevels = function () {
    var xs = [ 1, 2, [ 3, 4, [ 5, 6 ] ] ];
    assert.ok(!traverse.deepEqual(xs, []));
};
