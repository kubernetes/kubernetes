var acorn = require("../acorn.js");
var walk = require("./walk.js");
require("../test/codemirror-string.js");

var parsed = acorn.parse(codemirror30, {
  locations: true,
  ecmaVersion: 3,
  strictSemicolons: true,
  forbidReserved: true
});

walk.simple(parsed, {
  ScopeBody: function(node, scope) {
    node.scope = scope;
  }
}, walk.scopeVisitor);

var scopePasser = walk.make({
  ScopeBody: function(node, prev, c) { c(node, node.scope); }
});

var ignoredGlobals = Object.create(null);
"arguments window document navigator \
Array Math String Number RegExp Boolean Error Date Object \
setTimeout clearTimeout setInterval clearInterval \
Infinity NaN undefined JSON FileReader".split(" ").forEach(function(ignore) {
  ignoredGlobals[ignore] = true;
});
var hop = Object.prototype.hasOwnProperty;

function inScope(name, scope) {
  for (var cur = scope; cur; cur = cur.prev)
    if (hop.call(cur.vars, name)) return true;
}
function checkLHS(node, scope) {
  if (node.type == "Identifier" && !hop.call(ignoredGlobals, node.name) &&
      !inScope(node.name, scope)) {
    ignoredGlobals[node.name] = true;
    console.log("Assignment to global variable " + node.name +
                " (" + node.loc.start.line + ":" + node.loc.start.column + ")");
  }
}
function checkInScope(node, scope) {
  if (!hop.call(ignoredGlobals, node.name) && !inScope(node.name, scope)) {
    ignoredGlobals[node.name] = true;
    console.log("Using global variable " + node.name +
                " (" + node.loc.start.line + ":" + node.loc.start.column + ")");
  }
}

walk.simple(parsed, {
  UpdateExpression: function(node, scope) {checkLHS(node.argument, scope);},
  AssignmentExpression: function(node, scope) {checkLHS(node.left, scope);},
  Identifier: checkInScope
}, scopePasser);
