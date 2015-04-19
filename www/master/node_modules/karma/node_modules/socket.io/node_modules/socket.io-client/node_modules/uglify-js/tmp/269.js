var jsp = require("uglify-js").parser;
var pro = require("uglify-js").uglify;

var test_code = "var JSON;JSON||(JSON={});";

var ast = jsp.parse(test_code, false, false);
var nonembed_token_code = pro.gen_code(ast);
ast = jsp.parse(test_code, false, true);
var embed_token_code = pro.gen_code(ast);

console.log("original: " + test_code);
console.log("no token: " + nonembed_token_code);
console.log(" token: " + embed_token_code);
