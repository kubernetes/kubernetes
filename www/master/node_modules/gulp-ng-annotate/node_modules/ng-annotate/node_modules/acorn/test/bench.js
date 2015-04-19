require("./jquery-string.js");
require("./codemirror-string.js");

function test(lib, name) {
  for (var i =0, t0 = +new Date; i < 60; ++i) lib.parse(codemirror30);
  console.log(name + ": " + (+new Date - t0) + "ms");
}

/*test(require("./compare/esprima.js"), "Esprima");
test(require("../acorn.js"), "Acorn");
test(require("./compare/esprima.js"), "Esprima");
test(require("../acorn.js"), "Acorn");
test(require("./compare/esprima.js"), "Esprima");*/
test(require("../acorn.js"), "Acorn");
test(require("../acorn.js"), "Acorn");
test(require("../acorn.js"), "Acorn");

