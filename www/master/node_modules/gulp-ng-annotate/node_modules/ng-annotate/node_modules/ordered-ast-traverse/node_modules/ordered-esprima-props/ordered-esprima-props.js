// ordered-esprima-props.js
// MIT licensed, see LICENSE file
// Copyright (c) 2014 Olov Lassus <olov.lassus@gmail.com>

// A map from type (string) to an array of property names (strings)
// in lexical order, i.e. an AST-traversal in this order will visit
// nodes in increasing source code position.
//
// Tested with Esprima but should work for any Mozilla Parser API
// compatible AST, see
// https://developer.mozilla.org/en-US/docs/Mozilla/Projects/SpiderMonkey/Parser_API

module.exports = (function() {
    "use strict";

    function id(n) {
        return n;
    }

    var Expression = id;
    var Expressions = id;
    var Statement = id;
    var Statements = id;
    var Identifier = id;
    var Identifiers = id;
    var Lvalue = id;
    var BlockStatement = id;
    var CatchClause = id;
    var CatchClauses_singlearray = id;
    var VariableDeclarationOrIdentifier = id;
    var VariableDeclarationOrExpression = id;
    var VariableDeclarators = id;
    var Properties = id;
    var SwitchCases = id;

    return {
        ArrayExpression: [Expressions("elements")],
        AssignmentExpression: [Lvalue("left"), Expression("right")],
        BinaryExpression: [Expression("left"), Expression("right")],
        BlockStatement: [Statements("body")],
        BreakStatement: [Identifier("label")],
        CallExpression: [Expression("callee"), Expressions("arguments")],
        CatchClause: [Identifier("param"), /*"guard",*/ BlockStatement("body")],
        ConditionalExpression: [Expression("test"), Expression("consequent"), Expression("alternate")],
        ContinueStatement: [Identifier("label")],
        DebuggerStatement: [],
        DoWhileStatement: [Statement("body"), Expression("test")],
        EmptyStatement: [],
        ExpressionStatement: [Expression("expression")],
        ForInStatement: [VariableDeclarationOrIdentifier("left"), Expression("right"), Statement("body")],
        ForStatement: [VariableDeclarationOrExpression("init"), Expression("test"), Expression("update"), Statement("body")],
        FunctionDeclaration: [Identifier("id"), Identifiers("params"), BlockStatement("body")],
        FunctionExpression: [Identifier("id"), Identifiers("params"), BlockStatement("body")],
        Identifier: [],
        IfStatement: [Expression("test"), Statement("consequent"), Statement("alternate")],
        LabeledStatement: [Identifier("label"), Statement("body")],
        Literal: [],
        LogicalExpression: [Expression("left"), Expression("right")],
        MemberExpression: [Expression("object"), Identifier("property")],
        NewExpression: [Expression("callee"), Identifiers("arguments")],
        ObjectExpression: [Properties("properties")],
        Program: [Statements("body")],
        Property: [Identifier("key"), Expression("value")],
        ReturnStatement: [Expression("argument")],
        SequenceExpression: [Expressions("expressions")],
        SwitchCase: [Expression("test"), Statements("consequent")],
        SwitchStatement: [Expression("discriminant"), SwitchCases("cases")],
        ThisExpression: [],
        ThrowStatement: [Expression("argument")],
        TryStatement: [BlockStatement("block"), /*standard, acorn*/ CatchClause("handler"), /*esprima*/ CatchClauses_singlearray("handlers"), /*"guardedHandlers",*/ BlockStatement("finalizer")],
        UnaryExpression: [Expression("argument")],
        UpdateExpression: [Lvalue("argument")],
        VariableDeclaration: [VariableDeclarators("declarations")],
        VariableDeclarator: [Identifier("id"), Expression("init")],
        WhileStatement: [Expression("test"), Statement("body")],
        WithStatement: [Expression("object"), Statement("body")],
    };
})();
