// pos-to-linecolumn.js
// MIT licensed, see LICENSE file
// Copyright (c) 2014-2015 Olov Lassus <olov.lassus@gmail.com>

"use strict";

const assert = require("assert");

module.exports = PosToLineColumn;

function PosToLineColumn(str) {
    if (!(this instanceof PosToLineColumn)) {
        throw new Error("PosToLineColumn requires new");
    }
    str = String(str);

    const newlines = [];
    let pos = -1;
    while ((pos = str.indexOf("\n", pos + 1)) >= 0) {
        newlines.push(pos);
    }

    let line = 1;
    let column = 0;
    const columns = [];
    const lines = [];
    let i;
    let j = 0;
    for (i = 0; i < str.length; i++) {
        columns[i] = column;
        lines[i] = line;

        if (i === newlines[j]) {
            ++j;
            ++line;
            column = 0;
        } else {
            ++column;
        }
    }

    // add extra entry to support pos === str.length
    columns[i] = column;
    lines[i] = line;

    this.len = str.length;
    this.columns = columns;
    this.lines = lines;
}

PosToLineColumn.prototype.toLine = function(pos) {
    assert(pos >= 0 && pos <= this.len);
    return this.lines[pos];
};

PosToLineColumn.prototype.toColumn = function(pos) {
    assert(pos >= 0 && pos <= this.len);
    return this.columns[pos];
};

PosToLineColumn.prototype.toLineColumn = function(pos) {
    return {
        line: this.toLine(pos),
        column: this.toColumn(pos),
    };
};


/*
const tst = "asdf\n" +
    "abc\n" +
    "d\n" +
    "\n\n" +
    "efghi a\r\n" +
    "x";
const instance = new PosToLineColumn(tst);
console.dir(instance.toLineColumn(0));
console.dir(instance.toLineColumn(tst.length));
*/
