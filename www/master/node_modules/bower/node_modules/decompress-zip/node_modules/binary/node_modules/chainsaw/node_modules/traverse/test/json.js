var assert = require('assert');
var Traverse = require('traverse');

exports['json test'] = function () {
    var id = 54;
    var callbacks = {};
    var obj = { moo : function () {}, foo : [2,3,4, function () {}] };
    
    var scrubbed = Traverse(obj).map(function (x) {
        if (typeof x === 'function') {
            callbacks[id] = { id : id, f : x, path : this.path };
            this.update('[Function]');
            id++;
        }
    });
    
    assert.equal(
        scrubbed.moo, '[Function]',
        'obj.moo replaced with "[Function]"'
    );
    
    assert.equal(
        scrubbed.foo[3], '[Function]',
        'obj.foo[3] replaced with "[Function]"'
    );
    
    assert.deepEqual(scrubbed, {
        moo : '[Function]',
        foo : [ 2, 3, 4, "[Function]" ]
    }, 'Full JSON string matches');
    
    assert.deepEqual(
        typeof obj.moo, 'function',
        'Original obj.moo still a function'
    );
    
    assert.deepEqual(
        typeof obj.foo[3], 'function',
        'Original obj.foo[3] still a function'
    );
    
    assert.deepEqual(callbacks, {
        54: { id: 54, f : obj.moo, path: [ 'moo' ] },
        55: { id: 55, f : obj.foo[3], path: [ 'foo', '3' ] },
    }, 'Check the generated callbacks list');
};

