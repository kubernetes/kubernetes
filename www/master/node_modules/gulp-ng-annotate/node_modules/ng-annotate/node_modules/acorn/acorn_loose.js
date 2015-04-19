// Acorn: Loose parser
//
// This module provides an alternative parser (`parse_dammit`) that
// exposes that same interface as `parse`, but will try to parse
// anything as JavaScript, repairing syntax error the best it can.
// There are circumstances in which it will raise an error and give
// up, but they are very rare. The resulting AST will be a mostly
// valid JavaScript AST (as per the [Mozilla parser API][api], except
// that:
//
// - Return outside functions is allowed
//
// - Label consistency (no conflicts, break only to existing labels)
//   is not enforced.
//
// - Bogus Identifier nodes with a name of `"✖"` are inserted whenever
//   the parser got too confused to return anything meaningful.
//
// [api]: https://developer.mozilla.org/en-US/docs/SpiderMonkey/Parser_API
//
// The expected use for this is to *first* try `acorn.parse`, and only
// if that fails switch to `parse_dammit`. The loose parser might
// parse badly indented code incorrectly, so **don't** use it as
// your default parser.
//
// Quite a lot of acorn.js is duplicated here. The alternative was to
// add a *lot* of extra cruft to that file, making it less readable
// and slower. Copying and editing the code allowed me to make
// invasive changes and simplifications without creating a complicated
// tangle.

(function(root, mod) {
  if (typeof exports == "object" && typeof module == "object") return mod(exports, require("./acorn")); // CommonJS
  if (typeof define == "function" && define.amd) return define(["exports", "./acorn"], mod); // AMD
  mod(root.acorn || (root.acorn = {}), root.acorn); // Plain browser env
})(this, function(exports, acorn) {
  "use strict";

  var tt = acorn.tokTypes;

  var options, input, fetchToken, context;

  acorn.defaultOptions.tabSize = 4;

  exports.parse_dammit = function(inpt, opts) {
    if (!opts) opts = {};
    input = String(inpt);
    if (/^#!.*/.test(input)) input = "//" + input.slice(2);
    fetchToken = acorn.tokenize(input, opts);
    options = fetchToken.options;
    sourceFile = options.sourceFile || null;
    context = [];
    nextLineStart = 0;
    ahead.length = 0;
    next();
    return parseTopLevel();
  };

  var lastEnd, token = {start: 0, end: 0}, ahead = [];
  var curLineStart, nextLineStart, curIndent, lastEndLoc, sourceFile;

  function next(forceRegexp) {
    lastEnd = token.end;
    if (options.locations)
      lastEndLoc = token.endLoc;
    if (forceRegexp)
      ahead.length = 0;

    token = ahead.shift() || readToken(forceRegexp);

    if (token.start >= nextLineStart) {
      while (token.start >= nextLineStart) {
        curLineStart = nextLineStart;
        nextLineStart = lineEnd(curLineStart) + 1;
      }
      curIndent = indentationAfter(curLineStart);
    }
  }

  function readToken(forceRegexp) {
    for (;;) {
      try {
        var tok = fetchToken(forceRegexp);
        if (tok.type === tt.dot && input.substr(tok.end, 1) === '.') {
          tok = fetchToken();
          tok.start--;
          tok.type = tt.ellipsis;
        }
        return tok;
      } catch(e) {
        if (!(e instanceof SyntaxError)) throw e;

        // Try to skip some text, based on the error message, and then continue
        var msg = e.message, pos = e.raisedAt, replace = true;
        if (/unterminated/i.test(msg)) {
          pos = lineEnd(e.pos + 1);
          if (/string/.test(msg)) {
            replace = {start: e.pos, end: pos, type: tt.string, value: input.slice(e.pos + 1, pos)};
          } else if (/regular expr/i.test(msg)) {
            var re = input.slice(e.pos, pos);
            try { re = new RegExp(re); } catch(e) {}
            replace = {start: e.pos, end: pos, type: tt.regexp, value: re};
          } else if (/template/.test(msg)) {
            replace = {start: e.pos, end: pos,
                       type: input.charAt(e.pos) == "`" ? tt.template : tt.templateContinued,
                       value: input.slice(e.pos + 1, pos)};
          } else {
            replace = false;
          }
        } else if (/invalid (unicode|regexp|number)|expecting unicode|octal literal|is reserved|directly after number/i.test(msg)) {
          while (pos < input.length && !isSpace(input.charCodeAt(pos))) ++pos;
        } else if (/character escape|expected hexadecimal/i.test(msg)) {
          while (pos < input.length) {
            var ch = input.charCodeAt(pos++);
            if (ch === 34 || ch === 39 || isNewline(ch)) break;
          }
        } else if (/unexpected character/i.test(msg)) {
          pos++;
          replace = false;
        } else if (/regular expression/i.test(msg)) {
          replace = true;
        } else {
          throw e;
        }
        resetTo(pos);
        if (replace === true) replace = {start: pos, end: pos, type: tt.name, value: "✖"};
        if (replace) {
          if (options.locations) {
            replace.startLoc = acorn.getLineInfo(input, replace.start);
            replace.endLoc = acorn.getLineInfo(input, replace.end);
          }
          return replace;
        }
      }
    }
  }

  function resetTo(pos) {
    for (;;) {
      try {
        var ch = input.charAt(pos - 1);
        var reAllowed = !ch || /[\[\{\(,;:?\/*=+\-~!|&%^<>]/.test(ch) ||
          /[enwfd]/.test(ch) && /\b(keywords|case|else|return|throw|new|in|(instance|type)of|delete|void)$/.test(input.slice(pos - 10, pos));
        return fetchToken.jumpTo(pos, reAllowed);
      } catch(e) {
        if (!(e instanceof SyntaxError && /unterminated comment/i.test(e.message))) throw e;
        pos = lineEnd(e.pos + 1);
        if (pos >= input.length) return;
      }
    }
  }

  function lookAhead(n) {
    while (n > ahead.length)
      ahead.push(readToken());
    return ahead[n-1];
  }

  var newline = /[\n\r\u2028\u2029]/;

  function isNewline(ch) {
    return ch === 10 || ch === 13 || ch === 8232 || ch === 8329;
  }
  function isSpace(ch) {
    return (ch < 14 && ch > 8) || ch === 32 || ch === 160 || isNewline(ch);
  }

  function pushCx() {
    context.push(curIndent);
  }
  function popCx() {
    curIndent = context.pop();
  }

  function lineEnd(pos) {
    while (pos < input.length && !isNewline(input.charCodeAt(pos))) ++pos;
    return pos;
  }
  function indentationAfter(pos) {
    for (var count = 0;; ++pos) {
      var ch = input.charCodeAt(pos);
      if (ch === 32) ++count;
      else if (ch === 9) count += options.tabSize;
      else return count;
    }
  }

  function closes(closeTok, indent, line, blockHeuristic) {
    if (token.type === closeTok || token.type === tt.eof) return true;
    if (line != curLineStart && curIndent < indent && tokenStartsLine() &&
        (!blockHeuristic || nextLineStart >= input.length ||
         indentationAfter(nextLineStart) < indent)) return true;
    return false;
  }

  function tokenStartsLine() {
    for (var p = token.start - 1; p >= curLineStart; --p) {
      var ch = input.charCodeAt(p);
      if (ch !== 9 && ch !== 32) return false;
    }
    return true;
  }

  function Node(start) {
    this.type = null;
    this.start = start;
    this.end = null;
  }
  Node.prototype = acorn.Node.prototype;

  function SourceLocation(start) {
    this.start = start || token.startLoc || {line: 1, column: 0};
    this.end = null;
    if (sourceFile !== null) this.source = sourceFile;
  }

  function startNode() {
    var node = new Node(token.start);
    if (options.locations)
      node.loc = new SourceLocation();
    if (options.directSourceFile)
      node.sourceFile = options.directSourceFile;
    if (options.ranges)
      node.range = [token.start, 0];
    return node;
  }

  function storeCurrentPos() {
    return options.locations ? [token.start, token.startLoc] : token.start;
  }

  function startNodeAt(pos) {
    var node;
    if (options.locations) {
      node = new Node(pos[0]);
      node.loc = new SourceLocation(pos[1]);
    } else {
      node = new Node(pos);
    }
    if (options.directSourceFile)
      node.sourceFile = options.directSourceFile;
    if (options.ranges)
      node.range = [pos[0], 0];
    return node;
  }

  function finishNode(node, type) {
    node.type = type;
    node.end = lastEnd;
    if (options.locations)
      node.loc.end = lastEndLoc;
    if (options.ranges)
      node.range[1] = lastEnd;
    return node;
  }

  function finishNodeAt(node, type, pos) {
    if (options.locations) { node.loc.end = pos[1]; pos = pos[0]; }
    node.type = type;
    node.end = pos;
    if (options.ranges) node.range[1] = pos;
    return node;
  }

  function dummyIdent() {
    var dummy = startNode();
    dummy.name = "✖";
    return finishNode(dummy, "Identifier");
  }
  function isDummy(node) { return node.name == "✖"; }

  function eat(type) {
    if (token.type === type) {
      next();
      return true;
    } else {
      return false;
    }
  }

  function canInsertSemicolon() {
    return (token.type === tt.eof || token.type === tt.braceR || newline.test(input.slice(lastEnd, token.start)));
  }
  function semicolon() {
    return eat(tt.semi);
  }

  function expect(type) {
    if (eat(type)) return true;
    if (lookAhead(1).type == type) {
      next(); next();
      return true;
    }
    if (lookAhead(2).type == type) {
      next(); next(); next();
      return true;
    }
  }

  function checkLVal(expr) {
    if (!expr) return expr;
    switch (expr.type) {
      case "Identifier":
      case "MemberExpression":
      case "ObjectPattern":
      case "ArrayPattern":
      case "SpreadElement":
        return expr;

      default:
        return dummyIdent();
    }
  }

  function parseTopLevel() {
    var node = startNodeAt(options.locations ? [0, acorn.getLineInfo(input, 0)] : 0);
    node.body = [];
    while (token.type !== tt.eof) node.body.push(parseStatement());
    lastEnd = token.end;
    lastEndLoc = token.endLoc;
    return finishNode(node, "Program");
  }

  function parseStatement() {
    if (token.type === tt.slash || token.type === tt.assign && token.value === "/=")
      next(true);

    var starttype = token.type, node = startNode();

    switch (starttype) {
    case tt._break: case tt._continue:
      next();
      var isBreak = starttype === tt._break;
      if (semicolon() || canInsertSemicolon()) {
        node.label = null;
      } else {
        node.label = token.type === tt.name ? parseIdent() : null;
        semicolon();
      }
      return finishNode(node, isBreak ? "BreakStatement" : "ContinueStatement");

    case tt._debugger:
      next();
      semicolon();
      return finishNode(node, "DebuggerStatement");

    case tt._do:
      next();
      node.body = parseStatement();
      node.test = eat(tt._while) ? parseParenExpression() : dummyIdent();
      semicolon();
      return finishNode(node, "DoWhileStatement");

    case tt._for:
      next();
      pushCx();
      expect(tt.parenL);
      if (token.type === tt.semi) return parseFor(node, null);
      if (token.type === tt._var || token.type === tt._let) {
        var init = parseVar(true);
        if (init.declarations.length === 1 && (token.type === tt._in || token.type === tt.name && token.value === "of")) {
          return parseForIn(node, init);
        }
        return parseFor(node, init);
      }
      var init = parseExpression(false, true);
      if (token.type === tt._in || token.type === tt.name && token.value === "of") {
        return parseForIn(node, checkLVal(init));
      }
      return parseFor(node, init);

    case tt._function:
      next();
      return parseFunction(node, true);

    case tt._if:
      next();
      node.test = parseParenExpression();
      node.consequent = parseStatement();
      node.alternate = eat(tt._else) ? parseStatement() : null;
      return finishNode(node, "IfStatement");

    case tt._return:
      next();
      if (eat(tt.semi) || canInsertSemicolon()) node.argument = null;
      else { node.argument = parseExpression(); semicolon(); }
      return finishNode(node, "ReturnStatement");

    case tt._switch:
      var blockIndent = curIndent, line = curLineStart;
      next();
      node.discriminant = parseParenExpression();
      node.cases = [];
      pushCx();
      expect(tt.braceL);

      for (var cur; !closes(tt.braceR, blockIndent, line, true);) {
        if (token.type === tt._case || token.type === tt._default) {
          var isCase = token.type === tt._case;
          if (cur) finishNode(cur, "SwitchCase");
          node.cases.push(cur = startNode());
          cur.consequent = [];
          next();
          if (isCase) cur.test = parseExpression();
          else cur.test = null;
          expect(tt.colon);
        } else {
          if (!cur) {
            node.cases.push(cur = startNode());
            cur.consequent = [];
            cur.test = null;
          }
          cur.consequent.push(parseStatement());
        }
      }
      if (cur) finishNode(cur, "SwitchCase");
      popCx();
      eat(tt.braceR);
      return finishNode(node, "SwitchStatement");

    case tt._throw:
      next();
      node.argument = parseExpression();
      semicolon();
      return finishNode(node, "ThrowStatement");

    case tt._try:
      next();
      node.block = parseBlock();
      node.handler = null;
      if (token.type === tt._catch) {
        var clause = startNode();
        next();
        expect(tt.parenL);
        clause.param = parseIdent();
        expect(tt.parenR);
        clause.guard = null;
        clause.body = parseBlock();
        node.handler = finishNode(clause, "CatchClause");
      }
      node.finalizer = eat(tt._finally) ? parseBlock() : null;
      if (!node.handler && !node.finalizer) return node.block;
      return finishNode(node, "TryStatement");

    case tt._var:
    case tt._let:
    case tt._const:
      return parseVar();

    case tt._while:
      next();
      node.test = parseParenExpression();
      node.body = parseStatement();
      return finishNode(node, "WhileStatement");

    case tt._with:
      next();
      node.object = parseParenExpression();
      node.body = parseStatement();
      return finishNode(node, "WithStatement");

    case tt.braceL:
      return parseBlock();

    case tt.semi:
      next();
      return finishNode(node, "EmptyStatement");

    case tt._class:
      return parseObj(true, true);

    case tt._import:
      return parseImport();

    case tt._export:
      return parseExport();

    default:
      var expr = parseExpression();
      if (isDummy(expr)) {
        next();
        if (token.type === tt.eof) return finishNode(node, "EmptyStatement");
        return parseStatement();
      } else if (starttype === tt.name && expr.type === "Identifier" && eat(tt.colon)) {
        node.body = parseStatement();
        node.label = expr;
        return finishNode(node, "LabeledStatement");
      } else {
        node.expression = expr;
        semicolon();
        return finishNode(node, "ExpressionStatement");
      }
    }
  }

  function parseBlock() {
    var node = startNode();
    pushCx();
    expect(tt.braceL);
    var blockIndent = curIndent, line = curLineStart;
    node.body = [];
    while (!closes(tt.braceR, blockIndent, line, true))
      node.body.push(parseStatement());
    popCx();
    eat(tt.braceR);
    return finishNode(node, "BlockStatement");
  }

  function parseFor(node, init) {
    node.init = init;
    node.test = node.update = null;
    if (eat(tt.semi) && token.type !== tt.semi) node.test = parseExpression();
    if (eat(tt.semi) && token.type !== tt.parenR) node.update = parseExpression();
    popCx();
    expect(tt.parenR);
    node.body = parseStatement();
    return finishNode(node, "ForStatement");
  }

  function parseForIn(node, init) {
    var type = token.type === tt._in ? "ForInStatement" : "ForOfStatement";
    next();
    node.left = init;
    node.right = parseExpression();
    popCx();
    expect(tt.parenR);
    node.body = parseStatement();
    return finishNode(node, type);
  }

  function parseVar(noIn) {
    var node = startNode();
    node.kind = token.type.keyword;
    next();
    node.declarations = [];
    do {
      var decl = startNode();
      decl.id = options.ecmaVersion >= 6 ? toAssignable(parseExprAtom()) : parseIdent();
      decl.init = eat(tt.eq) ? parseExpression(true, noIn) : null;
      node.declarations.push(finishNode(decl, "VariableDeclarator"));
    } while (eat(tt.comma));
    if (!node.declarations.length) {
      var decl = startNode();
      decl.id = dummyIdent();
      node.declarations.push(finishNode(decl, "VariableDeclarator"));
    }
    if (!noIn) semicolon();
    return finishNode(node, "VariableDeclaration");
  }

  function parseExpression(noComma, noIn) {
    var start = storeCurrentPos();
    var expr = parseMaybeAssign(noIn);
    if (!noComma && token.type === tt.comma) {
      var node = startNodeAt(start);
      node.expressions = [expr];
      while (eat(tt.comma)) node.expressions.push(parseMaybeAssign(noIn));
      return finishNode(node, "SequenceExpression");
    }
    return expr;
  }

  function parseParenExpression() {
    pushCx();
    expect(tt.parenL);
    var val = parseExpression();
    popCx();
    expect(tt.parenR);
    return val;
  }

  function parseMaybeAssign(noIn) {
    var start = storeCurrentPos();
    var left = parseMaybeConditional(noIn);
    if (token.type.isAssign) {
      var node = startNodeAt(start);
      node.operator = token.value;
      node.left = token.type === tt.eq ? toAssignable(left) : checkLVal(left);
      next();
      node.right = parseMaybeAssign(noIn);
      return finishNode(node, "AssignmentExpression");
    }
    return left;
  }

  function parseMaybeConditional(noIn) {
    var start = storeCurrentPos();
    var expr = parseExprOps(noIn);
    if (eat(tt.question)) {
      var node = startNodeAt(start);
      node.test = expr;
      node.consequent = parseExpression(true);
      node.alternate = expect(tt.colon) ? parseExpression(true, noIn) : dummyIdent();
      return finishNode(node, "ConditionalExpression");
    }
    return expr;
  }

  function parseExprOps(noIn) {
    var start = storeCurrentPos();
    var indent = curIndent, line = curLineStart;
    return parseExprOp(parseMaybeUnary(noIn), start, -1, noIn, indent, line);
  }

  function parseExprOp(left, start, minPrec, noIn, indent, line) {
    if (curLineStart != line && curIndent < indent && tokenStartsLine()) return left;
    var prec = token.type.binop;
    if (prec != null && (!noIn || token.type !== tt._in)) {
      if (prec > minPrec) {
        var node = startNodeAt(start);
        node.left = left;
        node.operator = token.value;
        next();
        if (curLineStart != line && curIndent < indent && tokenStartsLine()) {
          node.right = dummyIdent();
        } else {
          var rightStart = storeCurrentPos();
          node.right = parseExprOp(parseMaybeUnary(noIn), rightStart, prec, noIn, indent, line);
        }
        finishNode(node, /&&|\|\|/.test(node.operator) ? "LogicalExpression" : "BinaryExpression");
        return parseExprOp(node, start, minPrec, noIn, indent, line);
      }
    }
    return left;
  }

  function parseMaybeUnary(noIn) {
    if (token.type.prefix) {
      var node = startNode(), update = token.type.isUpdate, nodeType;
      if (token.type === tt.ellipsis) {
        nodeType = "SpreadElement";
      } else {
        nodeType = update ? "UpdateExpression" : "UnaryExpression";
        node.operator = token.value;
        node.prefix = true;
      }
      node.operator = token.value;
      node.prefix = true;
      next();
      node.argument = parseMaybeUnary(noIn);
      if (update) node.argument = checkLVal(node.argument);
      return finishNode(node, nodeType);
    }
    var start = storeCurrentPos();
    var expr = parseExprSubscripts();
    while (token.type.postfix && !canInsertSemicolon()) {
      var node = startNodeAt(start);
      node.operator = token.value;
      node.prefix = false;
      node.argument = checkLVal(expr);
      next();
      expr = finishNode(node, "UpdateExpression");
    }
    return expr;
  }

  function parseExprSubscripts() {
    var start = storeCurrentPos();
    return parseSubscripts(parseExprAtom(), start, false, curIndent, curLineStart);
  }

  function parseSubscripts(base, start, noCalls, startIndent, line) {
    for (;;) {
      if (curLineStart != line && curIndent <= startIndent && tokenStartsLine()) {
        if (token.type == tt.dot && curIndent == startIndent)
          --startIndent;
        else
          return base;
      }

      if (eat(tt.dot)) {
        var node = startNodeAt(start);
        node.object = base;
        if (curLineStart != line && curIndent <= startIndent && tokenStartsLine())
          node.property = dummyIdent();
        else
          node.property = parsePropertyAccessor() || dummyIdent();
        node.computed = false;
        base = finishNode(node, "MemberExpression");
      } else if (token.type == tt.bracketL) {
        pushCx();
        next();
        var node = startNodeAt(start);
        node.object = base;
        node.property = parseExpression();
        node.computed = true;
        popCx();
        expect(tt.bracketR);
        base = finishNode(node, "MemberExpression");
      } else if (!noCalls && token.type == tt.parenL) {
        pushCx();
        var node = startNodeAt(start);
        node.callee = base;
        node.arguments = parseExprList(tt.parenR);
        base = finishNode(node, "CallExpression");
      } else if (token.type == tt.template) {
        var node = startNodeAt(start);
        node.tag = base;
        node.quasi = parseTemplate();
        base = finishNode(node, "TaggedTemplateExpression");
      } else {
        return base;
      }
    }
  }

  function parseExprAtom() {
    switch (token.type) {
    case tt._this:
      var node = startNode();
      next();
      return finishNode(node, "ThisExpression");

    case tt.name:
      var start = storeCurrentPos();
      var id = parseIdent();
      return eat(tt.arrow) ? parseArrowExpression(startNodeAt(start), [id]) : id;

    case tt.regexp:
      var node = startNode();
      var val = token.value;
      node.regex = {pattern: val.pattern, flags: val.flags};
      node.value = val.value;
      node.raw = input.slice(token.start, token.end);
      next();
      return finishNode(node, "Literal");

    case tt.num: case tt.string:
      var node = startNode();
      node.value = token.value;
      node.raw = input.slice(token.start, token.end);
      next();
      return finishNode(node, "Literal");

    case tt._null: case tt._true: case tt._false:
      var node = startNode();
      node.value = token.type.atomValue;
      node.raw = token.type.keyword;
      next();
      return finishNode(node, "Literal");

    case tt.parenL:
      var start = storeCurrentPos();
      next();
      var val = parseExpression();
      expect(tt.parenR);
      if (eat(tt.arrow)) {
        return parseArrowExpression(startNodeAt(start), val.expressions || (isDummy(val) ? [] : [val]));
      }
      if (options.preserveParens) {
        var par = startNodeAt(start);
        par.expression = val;
        val = finishNode(par, "ParenthesizedExpression");
      }
      return val;

    case tt.bracketL:
      var node = startNode();
      pushCx();
      node.elements = parseExprList(tt.bracketR, true);
      return finishNode(node, "ArrayExpression");

    case tt.braceL:
      return parseObj();

    case tt._class:
      return parseObj(true);

    case tt._function:
      var node = startNode();
      next();
      return parseFunction(node, false);

    case tt._new:
      return parseNew();

    case tt._yield:
      var node = startNode();
      next();
      if (semicolon() || canInsertSemicolon()) {
        node.delegate = false;
        node.argument = null;
      } else {
        node.delegate = eat(tt.star);
        node.argument = parseExpression(true);
      }
      return finishNode(node, "YieldExpression");

    case tt.template:
      return parseTemplate();

    default:
      return dummyIdent();
    }
  }

  function parseNew() {
    var node = startNode(), startIndent = curIndent, line = curLineStart;
    next();
    var start = storeCurrentPos();
    node.callee = parseSubscripts(parseExprAtom(), start, true, startIndent, line);
    if (token.type == tt.parenL) {
      pushCx();
      node.arguments = parseExprList(tt.parenR);
    } else {
      node.arguments = [];
    }
    return finishNode(node, "NewExpression");
  }

  function parseTemplateElement() {
    var elem = startNodeAt(options.locations ? [token.start + 1, token.startLoc.offset(1)] : token.start + 1);
    elem.value = token.value;
    elem.tail = input.charCodeAt(token.end - 1) !== 123; // '{'
    var endOff = elem.tail ? 1 : 2;
    var endPos = options.locations ? [token.end - endOff, token.endLoc.offset(-endOff)] : token.end - endOff;
    next();
    return finishNodeAt(elem, "TemplateElement", endPos);
  }

  function parseTemplate() {
    var node = startNode();
    node.expressions = [];
    var curElt = parseTemplateElement();
    node.quasis = [curElt];
    while (!curElt.tail) {
      var next = parseExpression();
      if (isDummy(next)) {
        node.quasis[node.quasis.length - 1].tail = true;
        break;
      }
      node.expressions.push(next);
      if (token.type === tt.templateContinued) {
        node.quasis.push(curElt = parseTemplateElement());
      } else {
        curElt = startNode();
        curElt.value = {cooked: "", raw: ""};
        curElt.tail = true;
        node.quasis.push(curElt);
      }
    }
    return finishNode(node, "TemplateLiteral");
  }

  function parseObj(isClass, isStatement) {
    var node = startNode();
    if (isClass) {
      next();
      if (token.type === tt.name) node.id = parseIdent();
      else if (isStatement) node.id = dummyIdent();
      node.superClass = eat(tt._extends) ? parseExpression() : null;
      node.body = startNode();
      node.body.body = [];
    } else {
      node.properties = [];
    }
    pushCx();
    var indent = curIndent + 1, line = curLineStart;
    eat(tt.braceL);
    if (curIndent + 1 < indent) { indent = curIndent; line = curLineStart; }
    while (!closes(tt.braceR, indent, line)) {
      var prop = startNode(), isGenerator;
      if (options.ecmaVersion >= 6) {
        if (isClass) {
          if (prop['static'] = (token.type === tt.name && token.value === "static")) next();
        } else {
          prop.method = false;
          prop.shorthand = false;
        }
        isGenerator = eat(tt.star);
      }
      parsePropertyName(prop);
      if (isDummy(prop.key)) { if (isDummy(parseExpression(true))) next(); eat(tt.comma); continue; }
      if (!isClass && eat(tt.colon)) {
        prop.kind = "init";
        prop.value = parseExpression(true);
      } else if (options.ecmaVersion >= 6 && (token.type === tt.parenL || token.type === tt.braceL)) {
        if (isClass) {
          prop.kind = "";
        } else {
          prop.kind = "init";
          prop.method = true;
        }
        prop.value = parseMethod(isGenerator);
      } else if (options.ecmaVersion >= 5 && prop.key.type === "Identifier" &&
                 (prop.key.name === "get" || prop.key.name === "set")) {
        prop.kind = prop.key.name;
        parsePropertyName(prop);
        prop.value = parseMethod(false);
      } else if (isClass) {
        prop.kind = "";
        prop.value = parseMethod(isGenerator);
      } else {
        prop.kind = "init";
        prop.value = options.ecmaVersion >= 6 ? prop.key : dummyIdent();
        prop.shorthand = true;
      }

      if (isClass) {
        node.body.body.push(finishNode(prop, "MethodDefinition"));
        semicolon();
      } else {
        node.properties.push(finishNode(prop, "Property"));
        eat(tt.comma);
      }
    }
    popCx();
    if (!eat(tt.braceR)) {
      // If there is no closing brace, make the node span to the start
      // of the next token (this is useful for Tern)
      lastEnd = token.start;
      if (options.locations) lastEndLoc = token.startLoc;
    }
    if (isClass) {
      semicolon();
      finishNode(node.body, "ClassBody");
      return finishNode(node, isStatement ? "ClassDeclaration" : "ClassExpression");
    } else {
      return finishNode(node, "ObjectExpression");
    }
  }

  function parsePropertyName(prop) {
    if (options.ecmaVersion >= 6) {
      if (eat(tt.bracketL)) {
        prop.computed = true;
        prop.key = parseExpression();
        expect(tt.bracketR);
        return;
      } else {
        prop.computed = false;
      }
    }
    var key = (token.type === tt.num || token.type === tt.string) ? parseExprAtom() : parseIdent();
    prop.key = key || dummyIdent();
  }

  function parsePropertyAccessor() {
    if (token.type === tt.name || token.type.keyword) return parseIdent();
  }

  function parseIdent() {
    var node = startNode();
    node.name = token.type === tt.name ? token.value : token.type.keyword;
    fetchToken.noRegexp();
    next();
    return finishNode(node, "Identifier");
  }

  function initFunction(node) {
    node.id = null;
    node.params = [];
    if (options.ecmaVersion >= 6) {
      node.defaults = [];
      node.rest = null;
      node.generator = false;
      node.expression = false;
    }
  }

  // Convert existing expression atom to assignable pattern
  // if possible.

  function toAssignable(node) {
    if (options.ecmaVersion >= 6 && node) {
      switch (node.type) {
        case "ObjectExpression":
          node.type = "ObjectPattern";
          var props = node.properties;
          for (var i = 0; i < props.length; i++) {
            props[i].value = toAssignable(props[i].value);
          }
          break;

        case "ArrayExpression":
          node.type = "ArrayPattern";
          var elms = node.elements;
          for (var i = 0; i < elms.length; i++) {
            elms[i] = toAssignable(elms[i]);
          }
          break;

        case "SpreadElement":
          node.argument = toAssignable(node.argument);
          break;
      }
    }
    return checkLVal(node);
  }

  function parseFunctionParams(node, params) {
    var defaults = [], hasDefaults = false;

    if (!params) {
      pushCx();
      params = parseExprList(tt.parenR);
    }
    for (var i = 0; i < params.length; i++) {
      var param = params[i], defValue = null;
      if (param.type === "AssignmentExpression") {
        defValue = param.right;
        param = param.left;
      }
      param = toAssignable(param);
      if (param.type === "SpreadElement") {
        param = param.argument;
        if (i === params.length - 1) {
          node.rest = param;
          continue;
        }
      }
      node.params.push(param);
      defaults.push(defValue);
      if (defValue) hasDefaults = true;
    }

    if (hasDefaults) node.defaults = defaults;
  }

  function parseFunction(node, isStatement) {
    initFunction(node);
    if (options.ecmaVersion >= 6) {
      node.generator = eat(tt.star);
    }
    if (token.type === tt.name) node.id = parseIdent();
    else if (isStatement) node.id = dummyIdent();
    parseFunctionParams(node);
    node.body = parseBlock();
    return finishNode(node, isStatement ? "FunctionDeclaration" : "FunctionExpression");
  }

  function parseMethod(isGenerator) {
    var node = startNode();
    initFunction(node);
    parseFunctionParams(node);
    node.generator = isGenerator || false;
    node.expression = options.ecmaVersion >= 6 && token.type !== tt.braceL;
    node.body = node.expression ? parseExpression(true) : parseBlock();
    return finishNode(node, "FunctionExpression");
  }

  function parseArrowExpression(node, params) {
    initFunction(node);
    parseFunctionParams(node, params);
    node.expression = token.type !== tt.braceL;
    node.body = node.expression ? parseExpression(true) : parseBlock();
    return finishNode(node, "ArrowFunctionExpression");
  }

  function parseExport() {
    var node = startNode();
    next();
    node['default'] = eat(tt._default);
    node.specifiers = node.source = null;
    if (node['default']) {
      node.declaration = parseExpression();
      semicolon();
    } else if (token.type.keyword) {
      node.declaration = parseStatement();
    } else {
      node.declaration = null;
      parseSpecifierList(node, "Export");
    }
    semicolon();
    return finishNode(node, "ExportDeclaration");
  }

  function parseImport() {
    var node = startNode();
    next();
    if (token.type === tt.string) {
      node.specifiers = [];
      node.source = parseExprAtom();
      node.kind = '';
    } else {
      if (token.type === tt.name && token.value !== "from") {
        var elt = startNode();
        elt.id = parseIdent();
        elt.name = null;
        elt['default'] = true;
        finishNode(elt, "ImportSpecifier");
        eat(tt.comma);
      }
      parseSpecifierList(node, "Import");
      var specs = node.specifiers;
      for (var i = 0; i < specs.length; i++) specs[i]['default'] = false;
      if (elt) node.specifiers.unshift(elt);
    }
    semicolon();
    return finishNode(node, "ImportDeclaration");
  }

  function parseSpecifierList(node, prefix) {
    var elts = node.specifiers = [];
    if (token.type === tt.star) {
      var elt = startNode();
      next();
      if (token.type === tt.name && token.value === "as") {
        next();
        elt.name = parseIdent();
      }
      elts.push(finishNode(elt, prefix + "BatchSpecifier"));
    } else {
      var indent = curIndent, line = curLineStart, continuedLine = nextLineStart;
      pushCx();
      eat(tt.braceL);
      if (curLineStart > continuedLine) continuedLine = curLineStart;
      while (!closes(tt.braceR, indent + (curLineStart <= continuedLine ? 1 : 0), line)) {
        var elt = startNode();
        if (token.type === tt.star) {
          next();
          if (token.type === tt.name && token.value === "as") {
            next();
            elt.name = parseIdent();
          }
          finishNode(elt, prefix + "BatchSpecifier");
        } else {
          if (token.type === tt.name && token.value === "from") break;
          elt.id = parseIdent();
          if (token.type === tt.name && token.value === "as") {
            next();
            elt.name = parseIdent();
          } else {
            elt.name = null;
          }
          finishNode(elt, prefix + "Specifier");
        }
        elts.push(elt);
        eat(tt.comma);
      }
      eat(tt.braceR);
      popCx();
    }
    if (token.type === tt.name && token.value === "from") {
      next();
      node.source = parseExprAtom();
    } else {
      node.source = null;
    }
  }

  function parseExprList(close, allowEmpty) {
    var indent = curIndent, line = curLineStart, elts = [];
    next(); // Opening bracket
    while (!closes(close, indent + 1, line)) {
      if (eat(tt.comma)) {
        elts.push(allowEmpty ? null : dummyIdent());
        continue;
      }
      var elt = parseExpression(true);
      if (isDummy(elt)) {
        if (closes(close, indent, line)) break;
        next();
      } else {
        elts.push(elt);
      }
      eat(tt.comma);
    }
    popCx();
    if (!eat(close)) {
      // If there is no closing brace, make the node span to the start
      // of the next token (this is useful for Tern)
      lastEnd = token.start;
      if (options.locations) lastEndLoc = token.startLoc;
    }
    return elts;
  }
});
