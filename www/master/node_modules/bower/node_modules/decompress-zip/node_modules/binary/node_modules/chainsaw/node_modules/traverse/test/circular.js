var assert = require('assert');
var Traverse = require('traverse');
var util = require('util');

exports.circular = function () {
    var obj = { x : 3 };
    obj.y = obj;
    var foundY = false;
    Traverse(obj).forEach(function (x) {
        if (this.path.join('') == 'y') {
            assert.equal(
                util.inspect(this.circular.node),
                util.inspect(obj)
            );
            foundY = true;
        }
    });
    assert.ok(foundY);
};

exports.deepCirc = function () {
    var obj = { x : [ 1, 2, 3 ], y : [ 4, 5 ] };
    obj.y[2] = obj;
    
    var times = 0;
    Traverse(obj).forEach(function (x) {
        if (this.circular) {
            assert.deepEqual(this.circular.path, []);
            assert.deepEqual(this.path, [ 'y', 2 ]);
            times ++;
        }
    });
    
    assert.deepEqual(times, 1);
};

exports.doubleCirc = function () {
    var obj = { x : [ 1, 2, 3 ], y : [ 4, 5 ] };
    obj.y[2] = obj;
    obj.x.push(obj.y);
    
    var circs = [];
    Traverse(obj).forEach(function (x) {
        if (this.circular) {
            circs.push({ circ : this.circular, self : this, node : x });
        }
    });
    
    assert.deepEqual(circs[0].self.path, [ 'x', 3, 2 ]);
    assert.deepEqual(circs[0].circ.path, []);
     
    assert.deepEqual(circs[1].self.path, [ 'y', 2 ]);
    assert.deepEqual(circs[1].circ.path, []);
    
    assert.deepEqual(circs.length, 2);
};

exports.circDubForEach = function () {
    var obj = { x : [ 1, 2, 3 ], y : [ 4, 5 ] };
    obj.y[2] = obj;
    obj.x.push(obj.y);
    
    Traverse(obj).forEach(function (x) {
        if (this.circular) this.update('...');
    });
    
    assert.deepEqual(obj, { x : [ 1, 2, 3, [ 4, 5, '...' ] ], y : [ 4, 5, '...' ] });
};

exports.circDubMap = function () {
    var obj = { x : [ 1, 2, 3 ], y : [ 4, 5 ] };
    obj.y[2] = obj;
    obj.x.push(obj.y);
    
    var c = Traverse(obj).map(function (x) {
        if (this.circular) {
            this.update('...');
        }
    });
    
    assert.deepEqual(c, { x : [ 1, 2, 3, [ 4, 5, '...' ] ], y : [ 4, 5, '...' ] });
};

exports.circClone = function () {
    var obj = { x : [ 1, 2, 3 ], y : [ 4, 5 ] };
    obj.y[2] = obj;
    obj.x.push(obj.y);
    
    var clone = Traverse.clone(obj);
    assert.ok(obj !== clone);
    
    assert.ok(clone.y[2] === clone);
    assert.ok(clone.y[2] !== obj);
    assert.ok(clone.x[3][2] === clone);
    assert.ok(clone.x[3][2] !== obj);
    assert.deepEqual(clone.x.slice(0,3), [1,2,3]);
    assert.deepEqual(clone.y.slice(0,2), [4,5]);
};

exports.circMapScrub = function () {
    var obj = { a : 1, b : 2 };
    obj.c = obj;
    
    var scrubbed = Traverse(obj).map(function (node) {
        if (this.circular) this.remove();
    });
    assert.deepEqual(
        Object.keys(scrubbed).sort(),
        [ 'a', 'b' ]
    );
    assert.ok(Traverse.deepEqual(scrubbed, { a : 1, b : 2 }));
    
    assert.equal(obj.c, obj);
};
