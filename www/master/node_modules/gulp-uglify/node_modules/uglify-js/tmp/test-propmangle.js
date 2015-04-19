#! /usr/bin/env node

var U = require("../tools/node.js");
var fs = require("fs");
var code = fs.readFileSync(process.argv[2], "utf8").replace(/^\s*#.*/, "");
var sys = require("util");

var ast = U.parse(code);

ast = U.mangle_properties(ast);

console.log("%s", ast.print_to_string({ beautify: true }));
