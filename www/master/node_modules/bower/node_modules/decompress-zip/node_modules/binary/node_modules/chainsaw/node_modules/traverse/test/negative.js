var Traverse = require('traverse');
var assert = require('assert');

exports['negative update test'] = function () {
    var obj = [ 5, 6, -3, [ 7, 8, -2, 1 ], { f : 10, g : -13 } ];
    var fixed = Traverse.map(obj, function (x) {
        if (x < 0) this.update(x + 128);
    });
    
    assert.deepEqual(fixed,
        [ 5, 6, 125, [ 7, 8, 126, 1 ], { f: 10, g: 115 } ],
        'Negative values += 128'
    );
    
    assert.deepEqual(obj,
        [ 5, 6, -3, [ 7, 8, -2, 1 ], { f: 10, g: -13 } ],
        'Original references not modified'
    );
}

