var assert = require('assert');
var Traverse = require('traverse');

exports.mutate = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse(obj).forEach(function (x) {
        if (typeof x === 'number' && x % 2 === 0) {
            this.update(x * 10);
        }
    });
    assert.deepEqual(obj, res);
    assert.deepEqual(obj, { a : 1, b : 20, c : [ 3, 40 ] });
};

exports.mutateT = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse.forEach(obj, function (x) {
        if (typeof x === 'number' && x % 2 === 0) {
            this.update(x * 10);
        }
    });
    assert.deepEqual(obj, res);
    assert.deepEqual(obj, { a : 1, b : 20, c : [ 3, 40 ] });
};

exports.map = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse(obj).map(function (x) {
        if (typeof x === 'number' && x % 2 === 0) {
            this.update(x * 10);
        }
    });
    assert.deepEqual(obj, { a : 1, b : 2, c : [ 3, 4 ] });
    assert.deepEqual(res, { a : 1, b : 20, c : [ 3, 40 ] });
};

exports.mapT = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse.map(obj, function (x) {
        if (typeof x === 'number' && x % 2 === 0) {
            this.update(x * 10);
        }
    });
    assert.deepEqual(obj, { a : 1, b : 2, c : [ 3, 4 ] });
    assert.deepEqual(res, { a : 1, b : 20, c : [ 3, 40 ] });
};

exports.clone = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse(obj).clone();
    assert.deepEqual(obj, res);
    assert.ok(obj !== res);
    obj.a ++;
    assert.deepEqual(res.a, 1);
    obj.c.push(5);
    assert.deepEqual(res.c, [ 3, 4 ]);
};

exports.cloneT = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse.clone(obj);
    assert.deepEqual(obj, res);
    assert.ok(obj !== res);
    obj.a ++;
    assert.deepEqual(res.a, 1);
    obj.c.push(5);
    assert.deepEqual(res.c, [ 3, 4 ]);
};

exports.reduce = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse(obj).reduce(function (acc, x) {
        if (this.isLeaf) acc.push(x);
        return acc;
    }, []);
    assert.deepEqual(obj, { a : 1, b : 2, c : [ 3, 4 ] });
    assert.deepEqual(res, [ 1, 2, 3, 4 ]);
};

exports.reduceInit = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse(obj).reduce(function (acc, x) {
        if (this.isRoot) assert.fail('got root');
        return acc;
    });
    assert.deepEqual(obj, { a : 1, b : 2, c : [ 3, 4 ] });
    assert.deepEqual(res, obj);
};

exports.remove = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    Traverse(obj).forEach(function (x) {
        if (this.isLeaf && x % 2 == 0) this.remove();
    });
    
    assert.deepEqual(obj, { a : 1, c : [ 3 ] });
};

exports.removeMap = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse(obj).map(function (x) {
        if (this.isLeaf && x % 2 == 0) this.remove();
    });
    
    assert.deepEqual(obj, { a : 1, b : 2, c : [ 3, 4 ] });
    assert.deepEqual(res, { a : 1, c : [ 3 ] });
};

exports.delete = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    Traverse(obj).forEach(function (x) {
        if (this.isLeaf && x % 2 == 0) this.delete();
    });
    
    assert.ok(!Traverse.deepEqual(
        obj, { a : 1, c : [ 3, undefined ] }
    ));
    
    assert.ok(Traverse.deepEqual(
        obj, { a : 1, c : [ 3 ] }
    ));
    
    assert.ok(!Traverse.deepEqual(
        obj, { a : 1, c : [ 3, null ] }
    ));
};

exports.deleteRedux = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4, 5 ] };
    Traverse(obj).forEach(function (x) {
        if (this.isLeaf && x % 2 == 0) this.delete();
    });
    
    assert.ok(!Traverse.deepEqual(
        obj, { a : 1, c : [ 3, undefined, 5 ] }
    ));
    
    assert.ok(Traverse.deepEqual(
        obj, { a : 1, c : [ 3 ,, 5 ] }
    ));
    
    assert.ok(!Traverse.deepEqual(
        obj, { a : 1, c : [ 3, null, 5 ] }
    ));
    
    assert.ok(!Traverse.deepEqual(
        obj, { a : 1, c : [ 3, 5 ] }
    ));
};

exports.deleteMap = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4 ] };
    var res = Traverse(obj).map(function (x) {
        if (this.isLeaf && x % 2 == 0) this.delete();
    });
    
    assert.ok(Traverse.deepEqual(
        obj,
        { a : 1, b : 2, c : [ 3, 4 ] }
    ));
    
    var xs = [ 3, 4 ];
    delete xs[1];
    
    assert.ok(Traverse.deepEqual(
        res, { a : 1, c : xs }
    ));
    
    assert.ok(Traverse.deepEqual(
        res, { a : 1, c : [ 3, ] }
    ));
    
    assert.ok(Traverse.deepEqual(
        res, { a : 1, c : [ 3 ] }
    ));
};

exports.deleteMapRedux = function () {
    var obj = { a : 1, b : 2, c : [ 3, 4, 5 ] };
    var res = Traverse(obj).map(function (x) {
        if (this.isLeaf && x % 2 == 0) this.delete();
    });
    
    assert.ok(Traverse.deepEqual(
        obj,
        { a : 1, b : 2, c : [ 3, 4, 5 ] }
    ));
    
    var xs = [ 3, 4, 5 ];
    delete xs[1];
    
    assert.ok(Traverse.deepEqual(
        res, { a : 1, c : xs }
    ));
    
    assert.ok(!Traverse.deepEqual(
        res, { a : 1, c : [ 3, 5 ] }
    ));
    
    assert.ok(Traverse.deepEqual(
        res, { a : 1, c : [ 3 ,, 5 ] }
    ));
};
