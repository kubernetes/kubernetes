var assert = require('assert');
var Traverse = require('traverse');

exports.stringify = function () {
    var obj = [ 5, 6, -3, [ 7, 8, -2, 1 ], { f : 10, g : -13 } ];
    
    var s = '';
    Traverse(obj).forEach(function (node) {
        if (Array.isArray(node)) {
            this.before(function () { s += '[' });
            this.post(function (child) {
                if (!child.isLast) s += ',';
            });
            this.after(function () { s += ']' });
        }
        else if (typeof node == 'object') {
            this.before(function () { s += '{' });
            this.pre(function (x, key) {
                s += '"' + key + '"' + ':';
            });
            this.post(function (child) {
                if (!child.isLast) s += ',';
            });
            this.after(function () { s += '}' });
        }
        else if (typeof node == 'function') {
            s += 'null';
        }
        else {
            s += node.toString();
        }
    });
    
    assert.equal(s, JSON.stringify(obj));
}

