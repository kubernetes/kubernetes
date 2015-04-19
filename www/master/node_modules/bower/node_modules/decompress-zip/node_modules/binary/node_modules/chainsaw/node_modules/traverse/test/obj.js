var assert = require('assert');
var Traverse = require('traverse');

exports['traverse an object with nested functions'] = function () {
    var to = setTimeout(function () {
        assert.fail('never ran');
    }, 1000);
    
    function Cons (x) {
        clearTimeout(to);
        assert.equal(x, 10);
    };
    Traverse(new Cons(10));
};

