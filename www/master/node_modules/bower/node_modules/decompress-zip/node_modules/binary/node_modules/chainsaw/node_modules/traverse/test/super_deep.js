var assert = require('assert');
var traverse = require('traverse');

exports.super_deep = function () {
    var util = require('util');
    var a0 = make();
    var a1 = make();
    assert.ok(traverse.deepEqual(a0, a1));
    
    a0.c.d.moo = true;
    assert.ok(!traverse.deepEqual(a0, a1));
    
    a1.c.d.moo = true;
    assert.ok(traverse.deepEqual(a0, a1));
    
    // TODO: this one
    //a0.c.a = a1;
    //assert.ok(!traverse.deepEqual(a0, a1));
};

function make () {
    var a = { self : 'a' };
    var b = { self : 'b' };
    var c = { self : 'c' };
    var d = { self : 'd' };
    var e = { self : 'e' };
    
    a.a = a;
    a.b = b;
    a.c = c;
    
    b.a = a;
    b.b = b;
    b.c = c;
    
    c.a = a;
    c.b = b;
    c.c = c;
    c.d = d;
    
    d.a = a;
    d.b = b;
    d.c = c;
    d.d = d;
    d.e = e;
    
    e.a = a;
    e.b = b;
    e.c = c;
    e.d = d;
    e.e = e;
    
    return a;
}
