// lut.js
// MIT licensed, see LICENSE file
// Copyright (c) 2013-2015 Olov Lassus <olov.lassus@gmail.com>

"use strict";

var assert = require("assert");
var traverse = require("ordered-ast-traverse");
var is = require("simple-is");

module.exports = Lut;

function Lut(ast, src) {
    assert(this instanceof Lut);

    var sparseBegins = new Array(src.length);
    var begins = [];
    var sparseEnds = new Array(src.length);
    var ends = [];
    var p = 0;
    var t0 = Date.now();
    traverse(ast, {pre: function(node) {
        //        assert (node.range[0] >= p);
        if (node.type === "Program") {
            return;
        }
        p = node.range[0];
        if (!sparseBegins[p]) {
            sparseBegins[p] = node;
        }
        p = node.range[1];
        if (!sparseEnds[p]) {
            sparseEnds[p] = node;
        }
    }});
    for (var i in sparseBegins) {
        begins.push(sparseBegins[i]);
    }
    for (var i$0 in sparseEnds) {
        ends.push(sparseEnds[i$0]);
    }
    var t1 = Date.now();
    //    console.error(t1-t0)

    // begins and ends are compact arrays with nodes,
    // sorted on node.range[0/1] (unique)
    this.begins = begins;
    this.ends = ends;
}

Lut.prototype.findNodeFromPos = findNodeFromPos;
Lut.prototype.findNodeBeforePos = findNodeBeforePos;

// binary search lut to find node beginning at pos
// or as close after pos as possible. null if none
function findNodeFromPos(pos) {
    var lut = this.begins;
    assert(is.finitenumber(pos) && pos >= 0);

    var left = 0;
    var right = lut.length - 1;
    while (left < right) {
        var mid = Math.floor((left + right) / 2);
        assert(mid >= 0 && mid < lut.length);
        if (pos > lut[mid].range[0]) {
            left = mid + 1;
        }
        else {
            right = mid;
        }
    }
    if (left > right) {
        assert(last(lut).range[0] < pos);
        return null;
    }

    var found = left;
    var foundPos = lut[found].range[0];
    assert(foundPos >= pos);
    if (found >= 1) {
        var prevPos = lut[found - 1].range[0];
        assert(prevPos < pos);
    }

    return lut[found];
}

// binary search lut to find node ending (as in range[1]
// at or before pos. null if none
function findNodeBeforePos(pos) {
    var lut = this.ends;
    assert(is.finitenumber(pos) && pos >= 0);

    var left = 0;
    var right = lut.length - 1;
    while (left < right) {
        var mid = Math.ceil((left + right) / 2);
        assert(mid >= 0 && mid < lut.length);
        if (pos < lut[mid].range[1]) {
            right = mid - 1;
        }
        else {
            left = mid;
        }
    }
    if (left > right) {
        assert(lut[0].range[1] > pos);
        return null;
    }

    var found = left;
    var foundPos = lut[found].range[1];
    if(foundPos > pos) {
        return null;
    }
    if (found <= lut.length - 2) {
        var nextPos = lut[found + 1].range[1];
        assert(nextPos > pos);
    }

    return lut[found];
}

function last(arr) {
    return arr[arr.length - 1];
}
