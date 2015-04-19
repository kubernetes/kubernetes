var assert = require('assert');
var Buffers = require('../');

function create (xs, split) {
    var bufs = Buffers();
    var offset = 0;
    split.forEach(function (i) {
        bufs.push(new Buffer(xs.slice(offset, offset + i)));
        offset += i;
    });
    return bufs;
}

exports.slice = function () {
    var xs = [0,1,2,3,4,5,6,7,8,9];
    var splits = [ [4,2,3,1], [2,2,2,2,2], [1,6,3,1], [9,2], [10], [5,5] ];
    
    splits.forEach(function (split) {
        var bufs = create(xs, split);
        assert.eql(new Buffer(xs), bufs.slice(),
            '[' + xs.join(',') + ']'
                + ' != ' + 
            '[' + [].join.call(bufs.slice(), ',') + ']'
        );
        
        for (var i = 0; i < xs.length; i++) {
            for (var j = i; j < xs.length; j++) {
                var a = bufs.slice(i,j);
                var b = new Buffer(xs.slice(i,j));
                
                assert.eql(a, b,
                    '[' + [].join.call(a, ',') + ']'
                        + ' != ' + 
                    '[' + [].join.call(b, ',') + ']'
                );
            }
        }
    });
};

exports.splice = function () {
    var xs = [0,1,2,3,4,5,6,7,8,9];
    var splits = [ [4,2,3,1], [2,2,2,2,2], [1,6,3,1], [9,2], [10], [5,5] ];
    
    splits.forEach(function (split) {
        for (var i = 0; i < xs.length; i++) {
            for (var j = i; j < xs.length; j++) {
                var bufs = create(xs, split);
                var xs_ = xs.slice();
                
                var a_ = bufs.splice(i,j);
                var a = [].slice.call(a_.slice());
                var b = xs_.splice(i,j);
                assert.eql(a, b,
                    '[' + a.join(',') + ']'
                        + ' != ' + 
                    '[' + b.join(',') + ']'
                );
                
                assert.eql(bufs.slice(), new Buffer(xs_),
                    '[' + [].join.call(bufs.slice(), ',') + ']'
                        + ' != ' + 
                    '[' + [].join.call(xs_, ',') + ']'
                );
            }
        }
    });
};

exports.spliceRep = function () {
    var xs = [0,1,2,3,4,5,6,7,8,9];
    var splits = [ [4,2,3,1], [2,2,2,2,2], [1,6,3,1], [9,2], [10], [5,5] ];
    var reps = [ [], [1], [5,6], [3,1,3,3,7], [9,8,7,6,5,4,3,2,1,2,3,4,5] ];
    
    splits.forEach(function (split) {
        reps.forEach(function (rep) {
            for (var i = 0; i < xs.length; i++) {
                for (var j = i; j < xs.length; j++) {
                    var bufs = create(xs, split);
                    var xs_ = xs.slice();
                    
                    var a_ = bufs.splice.apply(
                        bufs, [ i, j ].concat(new Buffer(rep))
                    );
                    var a = [].slice.call(a_.slice());
                    var b = xs_.splice.apply(xs_, [ i, j ].concat(rep));
                    
                    assert.eql(a, b,
                        '[' + a.join(',') + ']'
                            + ' != ' + 
                        '[' + b.join(',') + ']'
                    );
                    
                    assert.eql(bufs.slice(), new Buffer(xs_),
                        '[' + [].join.call(bufs.slice(), ',') + ']'
                            + ' != ' + 
                        '[' + [].join.call(xs_, ',') + ']'
                    );
                }
            }
        });
    });
}; 

exports.copy = function () {
    var xs = [0,1,2,3,4,5,6,7,8,9];
    var splits = [ [4,2,3,1], [2,2,2,2,2], [1,6,3,1], [9,2], [10], [5,5] ];
    
    splits.forEach(function (split) {
        var bufs = create(xs, split);
        var buf = new Buffer(xs);
        
        for (var i = 0; i < xs.length; i++) {
            for (var j = i; j < xs.length; j++) {
                var t0 = new Buffer(j - i);
                var t1 = new Buffer(j - i);
                
                assert.eql(
                    bufs.copy(t0, 0, i, j),
                    buf.copy(t1, 0, i, j)
                );
                
                assert.eql(
                    [].slice.call(t0),
                    [].slice.call(t1)
                );
            }
        }
    });
};

exports.push = function () {
    var bufs = Buffers();
    bufs.push(new Buffer([0]));
    bufs.push(new Buffer([1,2,3]));
    bufs.push(new Buffer([4,5]));
    bufs.push(new Buffer([6,7,8,9]));
    assert.eql(
        [].slice.call(bufs.slice()),
        [0,1,2,3,4,5,6,7,8,9]
    );
    
    assert.throws(function () {
        bufs.push(new Buffer([11,12]), 'moo');
    });
    assert.eql(bufs.buffers.length, 4);
};

exports.unshift = function () {
    var bufs = Buffers();
    bufs.unshift(new Buffer([6,7,8,9]));
    bufs.unshift(new Buffer([4,5]));
    bufs.unshift(new Buffer([1,2,3]));
    bufs.unshift(new Buffer([0]));
    assert.eql(
        [].slice.call(bufs.slice()),
        [0,1,2,3,4,5,6,7,8,9]
    );
    assert.throws(function () {
        bufs.unshift(new Buffer([-2,-1]), 'moo');
    });
    assert.eql(bufs.buffers.length, 4);
};

exports.get = function () {
    var bufs = Buffers();
    bufs.unshift(new Buffer([6,7,8,9]));
    bufs.unshift(new Buffer([4,5]));
    bufs.unshift(new Buffer([1,2,3]));
    bufs.unshift(new Buffer([0]));
    assert.eql( bufs.get(0), 0 );
    assert.eql( bufs.get(1), 1 );
    assert.eql( bufs.get(2), 2 );
    assert.eql( bufs.get(3), 3 );
    assert.eql( bufs.get(4), 4 );
    assert.eql( bufs.get(5), 5 );
    assert.eql( bufs.get(6), 6 );
    assert.eql( bufs.get(7), 7 );
    assert.eql( bufs.get(8), 8 );
    assert.eql( bufs.get(9), 9 );
};

exports.set = function () {
    var bufs = Buffers();
    bufs.push(new Buffer("Hel"));
    bufs.push(new Buffer("lo"));
    bufs.push(new Buffer("!"));
    bufs.set(0, 'h'.charCodeAt(0) );
    bufs.set(3, 'L'.charCodeAt(0) );
    bufs.set(5, '.'.charCodeAt(0) );
    assert.eql( bufs.slice(0).toString(), 'helLo.' );
};

exports.indexOf = function () {
    var bufs = Buffers();
    bufs.push(new Buffer("Hel"));
    bufs.push(new Buffer("lo,"));
    bufs.push(new Buffer(" how are "));
    bufs.push(new Buffer("you"));
    bufs.push(new Buffer("?"));
    assert.eql( bufs.indexOf("Hello"), 0 );
    assert.eql( bufs.indexOf("Hello", 1), -1 );
    assert.eql( bufs.indexOf("ello"), 1 );
    assert.eql( bufs.indexOf("ello", 1), 1 );
    assert.eql( bufs.indexOf("ello", 2), -1 );
    assert.eql( bufs.indexOf("e"), 1 );
    assert.eql( bufs.indexOf("e", 2), 13 );
    assert.eql( bufs.indexOf(new Buffer([0x65]), 2), 13 );
};
