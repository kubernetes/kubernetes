#!/usr/bin/env node
var Traverse = require('traverse');

var obj = [ 'five', 6, -3, [ 7, 8, -2, 1 ], { f : 10, g : -13 } ];

var s = '';
Traverse(obj).forEach(function to_s (node) {
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
            to_s(key);
            s += ':';
        });
        this.post(function (child) {
            if (!child.isLast) s += ',';
        });
        this.after(function () { s += '}' });
    }
    else if (typeof node == 'string') {
        s += '"' + node.toString().replace(/"/g, '\\"') + '"';
    }
    else if (typeof node == 'function') {
        s += 'null';
    }
    else {
        s += node.toString();
    }
});

console.log('JSON.stringify: ' + JSON.stringify(obj));
console.log('this stringify: ' + s);
