#! /usr/bin/env node

var fs = require("fs");
var u2 = require("../tools/node.js");

var ret = u2.minify("/tmp/test.js", {
    //outSourceMap: "test.js.map",
    mangle: true,
    compress: false,
    output: {
        beautify: true,
    }
});

console.log(ret.code);
