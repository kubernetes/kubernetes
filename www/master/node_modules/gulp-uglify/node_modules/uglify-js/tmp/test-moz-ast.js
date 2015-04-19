#! /usr/bin/env node

var U = require("../tools/node.js");
var acorn = require("acorn");
var fs = require("fs");
var code = fs.readFileSync(process.argv[2], "utf8").replace(/^\s*#.*/, "");
var sys = require("util");

var DELAY = 1;

var moz_ast, my_ast;

moz_ast = time_it("parse_acorn", function(){
    return acorn.parse(code, { locations: true, trackComments: true });
});

// time_it("parse_uglify", function(){
//     return U.parse(code);
// });

my_ast = time_it("transform AST", function(){
    return U.AST_Node.from_mozilla_ast(moz_ast);
});
my_ast.figure_out_scope();
my_ast = my_ast.transform(U.Compressor());
my_ast.figure_out_scope();
my_ast.compute_char_frequency();
U.base54.sort();
my_ast.mangle_names();

sys.print(my_ast.print_to_string({ beautify: false }));

function time_it(name, cont) {
    var t1 = new Date().getTime();
    var ret = cont();
    var spent = new Date().getTime() - t1;
    sys.error(name + ": " + (spent/1000).toFixed(3));
    return ret;
};
