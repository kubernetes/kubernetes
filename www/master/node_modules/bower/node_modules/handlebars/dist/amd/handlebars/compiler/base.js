define(
  ["./parser","./ast","./helpers","../utils","exports"],
  function(__dependency1__, __dependency2__, __dependency3__, __dependency4__, __exports__) {
    "use strict";
    var parser = __dependency1__["default"];
    var AST = __dependency2__["default"];
    var Helpers = __dependency3__;
    var extend = __dependency4__.extend;

    __exports__.parser = parser;

    var yy = {};
    extend(yy, Helpers, AST);

    function parse(input) {
      // Just return if an already-compile AST was passed in.
      if (input.constructor === AST.ProgramNode) { return input; }

      parser.yy = yy;

      return parser.parse(input);
    }

    __exports__.parse = parse;
  });