var test = require('tape');
var stable = require('./stable.js');


function cmp(a, b) {
    if (a === b) return 0;
    if (a > b) return 1;
    return -1;
}

function gt(a, b) {
    return a > b;
}

function diff(a, b) {
    return a - b;
}

function objCmp(a, b) {
    return a.x > b.x;
}

test('always returns a new array', function(t) {
    var array;

    array = [];
    t.doesNotEqual(array, stable(array));

    array = [1];
    t.doesNotEqual(array, stable(array));

    array = [1, 2];
    t.doesNotEqual(array, stable(array));

    t.end();
});

test('in-place always returns the same array', function(t) {
    var array;

    array = [];
    t.equal(array, stable.inplace(array));

    array = [1];
    t.equal(array, stable.inplace(array));

    array = [1, 2];
    t.equal(array, stable.inplace(array));

    t.end();
});

test('basic sorting', function(t) {
    t.same(
        stable(["foo", "bar", "baz"]),
        ["bar", "baz", "foo"]
    );

    t.same(
        stable([9, 2, 10, 5, 4, 3, 0, 1, 8, 6, 7]),
        [0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
    );

    t.end();
});

test('comparators', function(t) {
    t.same(
        stable([9, 2, 10, 5, 4, 3, 0, 1, 8, 6, 7], cmp),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );

    t.same(
        stable([9, 2, 10, 5, 4, 3, 0, 1, 8, 6, 7], gt),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );

    t.same(
        stable([9, 2, 10, 5, 4, 3, 0, 1, 8, 6, 7], diff),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );

    t.same(
        stable([{x:4}, {x:3}, {x:5}], objCmp),
        [{x:3}, {x:4}, {x:5}]
    );

    t.end();
});

test('stable sorting', function(t) {
    function cmp(a, b) {
        return a.x > b.x;
    }
    t.same(
        stable([{x:3,y:1}, {x:4,y:2}, {x:3,y:3}, {x:5,y:4}, {x:3,y:5}], cmp),
        [{x:3,y:1}, {x:3,y:3}, {x:3,y:5}, {x:4,y:2}, {x:5,y:4}]
    );

    t.end();
});
