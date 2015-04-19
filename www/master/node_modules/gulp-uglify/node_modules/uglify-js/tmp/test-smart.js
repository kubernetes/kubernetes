#! /usr/bin/env node

var U = require("../tools/node.js");
var fs = require("fs");
var code = fs.readFileSync(process.argv[2], "utf8").replace(/^\s*#.*/, "");
var sys = require("util");

var ast = U.parse(code);

ast = ast.smart_normalize();
sys.print(ast.print_to_string({ beautify: true, ascii_only: true }));
sys.print("\n---\n");

var defun = null;
try { ast.walk(new U.TreeWalker(function(node){
    if (node instanceof U.AST_Lambda) throw defun = node;
})) } catch(ex){};

defun.smart_annotate_cfg([]);

defun.walk(new U.TreeWalker(function(node){
    if (node instanceof U.AST_Definitions) return true;
    if (!node.smart_info) {
        sys.print("*****", node.print_to_string({}), "*****");
    } else {
        sys.print(show(node) + " → [" + node.smart_info.next.map(show).join(" # ") + "] ← [" + node.smart_info.prev.map(show).join(" # ") + "]");
    }
    sys.print("\n");
}));

function show(node) {
    return node.print_to_string({ beautify: true }).split(/\n/)[0];
}









// ast.walk(new U.TreeWalker(function walk(node){
//     if (!node._done && (node instanceof U.AST_SymbolDeclaration || (node instanceof U.AST_SymbolRef && !node.smart_undeclared))) {
//         node._done = true;
//         var s = node.smart_scope;
//         var v = s.find_var(node);
//         if (v) {
//             node.name = U.string_template("{name}_{frame}_{index}", {
//                 name: node.name,
//                 frame: v.frame,
//                 index: v.index
//             });
//         }
//     }
// }));

// sys.print(ast.print_to_string({ beautify: true, ascii_only: true }));
