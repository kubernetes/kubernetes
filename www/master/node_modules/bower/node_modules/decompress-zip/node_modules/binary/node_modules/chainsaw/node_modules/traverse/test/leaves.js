var assert = require('assert');
var Traverse = require('traverse');

exports['leaves test'] = function () {
    var acc = [];
    Traverse({
        a : [1,2,3],
        b : 4,
        c : [5,6],
        d : { e : [7,8], f : 9 }
    }).forEach(function (x) {
        if (this.isLeaf) acc.push(x);
    });
    
    assert.equal(
        acc.join(' '),
        '1 2 3 4 5 6 7 8 9',
        'Traversal in the right(?) order'
    );
};

