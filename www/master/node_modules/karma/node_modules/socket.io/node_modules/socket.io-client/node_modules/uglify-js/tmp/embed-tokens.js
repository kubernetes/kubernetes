#! /usr/bin/env node

global.sys = require(/^v0\.[012]/.test(process.version) ? "sys" : "util");
var fs = require("fs");
var uglify = require("uglify-js"), // symlink ~/.node_libraries/uglify-js.js to ../uglify-js.js
    jsp = uglify.parser,
    pro = uglify.uglify;

var code = fs.readFileSync("embed-tokens.js", "utf8").replace(/^#.*$/mg, "");
var ast = jsp.parse(code, null, true);

// trololo
function fooBar() {}

console.log(sys.inspect(ast, null, null));
