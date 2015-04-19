#! /usr/bin/env node

var U2 = require("../tools/node.js");

var code = function moo(foo, bar) {
    OUT: if (foo) {
        if (bar) {
            break OUT;
        }
        console.log("WAT");
    }
}.toString();

var ast = U2.parse(code);

var clone = ast.transform(new U2.TreeTransformer(null, function(node){
    console.log(node.TYPE, node.print_to_string());
}));
var compressor = U2.Compressor();

console.log(clone.print_to_string());

ast.figure_out_scope();
clone.figure_out_scope();

var compressed = clone.transform(compressor);

console.log(compressed.print_to_string({ beautify: true }));
