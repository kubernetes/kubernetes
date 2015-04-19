var assert = require('assert');
var traverse = require('traverse');

exports.stop = function () {
    var visits = 0;
    traverse('abcdefghij'.split('')).forEach(function (node) {
        if (typeof node === 'string') {
            visits ++;
            if (node === 'e') this.stop()
        }
    });
    
    assert.equal(visits, 5);
};

exports.stopMap = function () {
    var s = traverse('abcdefghij'.split('')).map(function (node) {
        if (typeof node === 'string') {
            if (node === 'e') this.stop()
            return node.toUpperCase();
        }
    }).join('');
    
    assert.equal(s, 'ABCDEfghij');
};

exports.stopReduce = function () {
    var obj = {
        a : [ 4, 5 ],
        b : [ 6, [ 7, 8, 9 ] ]
    };
    var xs = traverse(obj).reduce(function (acc, node) {
        if (this.isLeaf) {
            if (node === 7) this.stop();
            else acc.push(node)
        }
        return acc;
    }, []);
    
    assert.deepEqual(xs, [ 4, 5, 6 ]);
};
