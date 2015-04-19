// ng-annotate-main.js
// MIT licensed, see LICENSE file
// Copyright (c) 2013-2015 Olov Lassus <olov.lassus@gmail.com>

"use strict";
const fmt = require("simple-fmt");
const is = require("simple-is");
const alter = require("alter");
const traverse = require("ordered-ast-traverse");
const EOL = require("os").EOL;
const assert = require("assert");
const ngInject = require("./nginject");
const generateSourcemap = require("./generate-sourcemap");
const Lut = require("./lut");
const scopeTools = require("./scopetools");
const stringmap = require("stringmap");
const require_acorn_t0 = Date.now();
const parser = require("acorn").parse;
const require_acorn_t1 = Date.now();

const chainedRouteProvider = 1;
const chainedUrlRouterProvider = 2;
const chainedStateProvider = 3;
const chainedRegular = 4;

function match(node, ctx, matchPlugins) {
    const isMethodCall = (
        node.type === "CallExpression" &&
            node.callee.type === "MemberExpression" &&
            node.callee.computed === false
        );

    // matchInjectorInvoke must happen before matchRegular
    // to prevent false positive ($injector.invoke() outside module)
    // matchProvide must happen before matchRegular
    // to prevent regular from matching it as a short-form
    const matchMethodCalls = (isMethodCall &&
        (matchInjectorInvoke(node) || matchProvide(node, ctx) || matchRegular(node, ctx) || matchNgRoute(node) || matchMaterialShowModalOpen(node) || matchNgUi(node) || matchHttpProvider(node)));

    return matchMethodCalls ||
        (matchPlugins && matchPlugins(node)) ||
        matchDirectiveReturnObject(node) ||
        matchProviderGet(node);
}

function matchMaterialShowModalOpen(node) {
    // $mdDialog.show({.. controller: fn, resolve: {f: function($scope) {}, ..}});
    // $mdToast.show({.. controller: fn, resolve: {f: function($scope) {}, ..}});
    // $mdBottomSheet.show({.. controller: fn, resolve: {f: function($scope) {}, ..}});
    // $modal.open({.. controller: fn, resolve: {f: function($scope) {}, ..}});

    // we already know that node is a (non-computed) method call
    const callee = node.callee;
    const obj = callee.object; // identifier or expression
    const method = callee.property; // identifier
    const args = node.arguments;

    if (obj.type === "Identifier" &&
        ((obj.name === "$modal" && method.name === "open") || (is.someof(obj.name, ["$mdDialog", "$mdToast", "$mdBottomSheet"]) && method.name === "show")) &&
        args.length === 1 && args[0].type === "ObjectExpression") {
        const props = args[0].properties;
        const res = [matchProp("controller", props)];
        res.push.apply(res, matchResolve(props));
        return res.filter(Boolean);
    }
    return false;
}

function matchDirectiveReturnObject(node) {
    // only matches inside directives
    // return { .. controller: function($scope, $timeout), ...}

    return limit("directive", node.type === "ReturnStatement" &&
        node.argument && node.argument.type === "ObjectExpression" &&
        matchProp("controller", node.argument.properties));
}

function limit(name, node) {
    if (node && !node.$limitToMethodName) {
        node.$limitToMethodName = name;
    }
    return node;
}

function matchProviderGet(node) {
    // only matches inside providers
    // (this|self|that).$get = function($scope, $timeout)
    // { ... $get: function($scope, $timeout), ...}
    let memberExpr;
    let self;
    return limit("provider", (node.type === "AssignmentExpression" && (memberExpr = node.left).type === "MemberExpression" &&
        memberExpr.property.name === "$get" &&
        ((self = memberExpr.object).type === "ThisExpression" || (self.type === "Identifier" && is.someof(self.name, ["self", "that"]))) &&
        node.right) ||
        (node.type === "ObjectExpression" && matchProp("$get", node.properties)));
}

function matchNgRoute(node) {
    // $routeProvider.when("path", {
    //   ...
    //   controller: function($scope) {},
    //   resolve: {f: function($scope) {}, ..}
    // })

    // we already know that node is a (non-computed) method call
    const callee = node.callee;
    const obj = callee.object; // identifier or expression
    if (!(obj.$chained === chainedRouteProvider || (obj.type === "Identifier" && obj.name === "$routeProvider"))) {
        return false;
    }
    node.$chained = chainedRouteProvider;

    const method = callee.property; // identifier
    if (method.name !== "when") {
        return false;
    }

    const args = node.arguments;
    if (args.length !== 2) {
        return false;
    }
    const configArg = last(args)
    if (configArg.type !== "ObjectExpression") {
        return false;
    }

    const props = configArg.properties;
    const res = [
        matchProp("controller", props)
    ];
    // {resolve: ..}
    res.push.apply(res, matchResolve(props));

    const filteredRes = res.filter(Boolean);
    return (filteredRes.length === 0 ? false : filteredRes);
}

function matchNgUi(node) {
    // $stateProvider.state("myState", {
    //     ...
    //     controller: function($scope)
    //     controllerProvider: function($scope)
    //     templateProvider: function($scope)
    //     onEnter: function($scope)
    //     onExit: function($scope)
    // });
    // $stateProvider.state("myState", {... resolve: {f: function($scope) {}, ..} ..})
    // $stateProvider.state("myState", {... views: {... somename: {... controller: fn, controllerProvider: fn, templateProvider: fn, resolve: {f: fn}}}})
    //
    // stateHelperProvider.setNestedState({ sameasregularstate, children: [sameasregularstate, ..]})
    // stateHelperProvider.setNestedState({ sameasregularstate, children: [sameasregularstate, ..]}, true)
    //
    // $urlRouterProvider.when(.., function($scope) {})
    //
    // $modal.open see matchMaterialShowModalOpen

    // we already know that node is a (non-computed) method call
    const callee = node.callee;
    const obj = callee.object; // identifier or expression
    const method = callee.property; // identifier
    const args = node.arguments;

    // shortcut for $urlRouterProvider.when(.., function($scope) {})
    if (obj.$chained === chainedUrlRouterProvider || (obj.type === "Identifier" && obj.name === "$urlRouterProvider")) {
        node.$chained = chainedUrlRouterProvider;

        if (method.name === "when" && args.length >= 1) {
            return last(args);
        }
        return false;
    }

    // everything below is for $stateProvider and stateHelperProvider alone
    if (!(obj.$chained === chainedStateProvider || (obj.type === "Identifier" && is.someof(obj.name, ["$stateProvider", "stateHelperProvider"])))) {
        return false;
    }
    node.$chained = chainedStateProvider;

    if (is.noneof(method.name, ["state", "setNestedState"])) {
        return false;
    }

    // $stateProvider.state({ ... }) and $stateProvider.state("name", { ... })
    // stateHelperProvider.setNestedState({ .. }) and stateHelperProvider.setNestedState({ .. }, true)
    if (!(args.length >= 1 && args.length <= 2)) {
        return false;
    }

    const configArg = (method.name === "state" ? last(args) : args[0]);

    const res = [];

    recursiveMatch(configArg);

    const filteredRes = res.filter(Boolean);
    return (filteredRes.length === 0 ? false : filteredRes);


    function recursiveMatch(objectExpressionNode) {
        if (!objectExpressionNode || objectExpressionNode.type !== "ObjectExpression") {
            return false;
        }

        const properties = objectExpressionNode.properties;

        matchStateProps(properties, res);

        const childrenArrayExpression = matchProp("children", properties);
        const children = childrenArrayExpression && childrenArrayExpression.elements;

        if (!children) {
            return;
        }
        children.forEach(recursiveMatch);
    }

    function matchStateProps(props, res) {
        const simple = [
            matchProp("controller", props),
            matchProp("controllerProvider", props),
            matchProp("templateProvider", props),
            matchProp("onEnter", props),
            matchProp("onExit", props),
        ];
        res.push.apply(res, simple);

        // {resolve: ..}
        res.push.apply(res, matchResolve(props));

        // {view: ...}
        const viewObject = matchProp("views", props);
        if (viewObject && viewObject.type === "ObjectExpression") {
            viewObject.properties.forEach(function(prop) {
                if (prop.value.type === "ObjectExpression") {
                    res.push(matchProp("controller", prop.value.properties));
                    res.push(matchProp("controllerProvider", prop.value.properties));
                    res.push(matchProp("templateProvider", prop.value.properties));
                    res.push.apply(res, matchResolve(prop.value.properties));
                }
            });
        }
    }
}

function matchInjectorInvoke(node) {
    // $injector.invoke(function($compile) { ... });

    // we already know that node is a (non-computed) method call
    const callee = node.callee;
    const obj = callee.object; // identifier or expression
    const method = callee.property; // identifier

    return method.name === "invoke" &&
        obj.type === "Identifier" && obj.name === "$injector" &&
        node.arguments.length >= 1 && node.arguments;
}

function matchHttpProvider(node) {
    // $httpProvider.interceptors.push(function($scope) {});
    // $httpProvider.responseInterceptors.push(function($scope) {});

    // we already know that node is a (non-computed) method call
    const callee = node.callee;
    const obj = callee.object; // identifier or expression
    const method = callee.property; // identifier

    return (method.name === "push" &&
        obj.type === "MemberExpression" && !obj.computed &&
        obj.object.name === "$httpProvider" && is.someof(obj.property.name,  ["interceptors", "responseInterceptors"]) &&
        node.arguments.length >= 1 && node.arguments);
}

function matchProvide(node, ctx) {
    // $provide.decorator("foo", function($scope) {});
    // $provide.service("foo", function($scope) {});
    // $provide.factory("foo", function($scope) {});
    // $provide.provider("foo", function($scope) {});

    // we already know that node is a (non-computed) method call
    const callee = node.callee;
    const obj = callee.object; // identifier or expression
    const method = callee.property; // identifier
    const args = node.arguments;

    const target = obj.type === "Identifier" && obj.name === "$provide" &&
        is.someof(method.name, ["decorator", "service", "factory", "provider"]) &&
        args.length === 2 && args[1];

    if (target) {
        target.$methodName = method.name;

        if (ctx.rename) {
            // for eventual rename purposes
            return args;
        }
    }
    return target;
}

function matchRegular(node, ctx) {
    // we already know that node is a (non-computed) method call
    const callee = node.callee;
    const obj = callee.object; // identifier or expression
    const method = callee.property; // identifier

    // short-cut implicit config special case:
    // angular.module("MyMod", function(a) {})
    if (obj.name === "angular" && method.name === "module") {
        const args = node.arguments;
        if (args.length >= 2) {
            node.$chained = chainedRegular;
            return last(args);
        }
    }

    const matchAngularModule = (obj.$chained === chainedRegular || isReDef(obj, ctx) || isLongDef(obj)) &&
        is.someof(method.name, ["provider", "value", "constant", "bootstrap", "config", "factory", "directive", "filter", "run", "controller", "service", "animation", "invoke"]);
    if (!matchAngularModule) {
        return false;
    }
    node.$chained = chainedRegular;

    if (is.someof(method.name, ["value", "constant", "bootstrap"])) {
        return false; // affects matchAngularModule because of chaining
    }

    const args = node.arguments;
    const target = (is.someof(method.name, ["config", "run"]) ?
        args.length === 1 && args[0] :
        args.length === 2 && args[0].type === "Literal" && is.string(args[0].value) && args[1]);

    if (target) {
        target.$methodName = method.name;
    }

    if (ctx.rename && args.length === 2 && target) {
        // for eventual rename purposes
        const somethingNameLiteral = args[0];
        return [somethingNameLiteral, target];
    }
    return target;
}

// matches with default regexp
//   *.controller("MyCtrl", function($scope, $timeout) {});
//   *.*.controller("MyCtrl", function($scope, $timeout) {});
// matches with --regexp "^require(.*)$"
//   require("app-module").controller("MyCtrl", function($scope) {});
function isReDef(node, ctx) {
    return ctx.re.test(ctx.srcForRange(node.range));
}

// Long form: angular.module(*).controller("MyCtrl", function($scope, $timeout) {});
function isLongDef(node) {
    return node.callee &&
        node.callee.object && node.callee.object.name === "angular" &&
        node.callee.property && node.callee.property.name === "module";
}

function last(arr) {
    return arr[arr.length - 1];
}

function matchProp(name, props) {
    for (let i = 0; i < props.length; i++) {
        const prop = props[i];
        if ((prop.key.type === "Identifier" && prop.key.name === name) ||
            (prop.key.type === "Literal" && prop.key.value === name)) {
            return prop.value; // FunctionExpression or ArrayExpression
        }
    }
    return null;
}

function matchResolve(props) {
    const resolveObject = matchProp("resolve", props);
    if (resolveObject && resolveObject.type === "ObjectExpression") {
        return resolveObject.properties.map(function(prop) {
            return prop.value;
        });
    }
    return [];
};

function renamedString(ctx, originalString) {
    if (ctx.rename) {
        return ctx.rename.get(originalString) || originalString;
    }
    return originalString;
}

function stringify(ctx, arr, quot) {
    return "[" + arr.map(function(arg) {
        return quot + renamedString(ctx, arg.name) + quot;
    }).join(", ") + "]";
}

function parseExpressionOfType(str, type) {
    const node = parser(str).body[0].expression;
    assert(node.type === type);
    return node;
}

// stand-in for not having a jsshaper-style ref's
function replaceNodeWith(node, newNode) {
    let done = false;
    const parent = node.$parent;
    const keys = Object.keys(parent);
    keys.forEach(function(key) {
        if (parent[key] === node) {
            parent[key] = newNode;
            done = true;
        }
    });

    if (done) {
        return;
    }

    // second pass, now check arrays
    keys.forEach(function(key) {
        if (Array.isArray(parent[key])) {
            const arr = parent[key];
            for (let i = 0; i < arr.length; i++) {
                if (arr[i] === node) {
                    arr[i] = newNode;
                    done = true;
                }
            }
        }
    });

    assert(done);
}

function insertArray(ctx, functionExpression, fragments, quot) {
    const args = stringify(ctx, functionExpression.params, quot);

    fragments.push({
        start: functionExpression.range[0],
        end: functionExpression.range[0],
        str: args.slice(0, -1) + ", ",
        loc: {
            start: functionExpression.loc.start,
            end: functionExpression.loc.start
        }
    });
    fragments.push({
        start: functionExpression.range[1],
        end: functionExpression.range[1],
        str: "]",
        loc: {
            start: functionExpression.loc.end,
            end: functionExpression.loc.end
        }
    });
}

function replaceArray(ctx, array, fragments, quot) {
    const functionExpression = last(array.elements);

    if (functionExpression.params.length === 0) {
        return removeArray(array, fragments);
    }

    const args = stringify(ctx, functionExpression.params, quot);
    fragments.push({
        start: array.range[0],
        end: functionExpression.range[0],
        str: args.slice(0, -1) + ", ",
        loc: {
            start: array.loc.start,
            end: functionExpression.loc.start
        }
    });
}

function removeArray(array, fragments) {
    const functionExpression = last(array.elements);

    fragments.push({
        start: array.range[0],
        end: functionExpression.range[0],
        str: "",
        loc: {
            start: array.loc.start,
            end: functionExpression.loc.start
        }
    });
    fragments.push({
        start: functionExpression.range[1],
        end: array.range[1],
        str: "",
        loc: {
            start: functionExpression.loc.end,
            end: array.loc.end
        }
    });
}

function renameProviderDeclarationSite(ctx, literalNode, fragments) {
    fragments.push({
        start: literalNode.range[0] + 1,
        end: literalNode.range[1] - 1,
        str: renamedString(ctx, literalNode.value),
        loc: {
            start: {
                line: literalNode.loc.start.line,
                column: literalNode.loc.start.column + 1
            }, end: {
                line: literalNode.loc.end.line,
                column: literalNode.loc.end.column - 1
            }
        }
    });
}

function judgeSuspects(ctx) {
    const mode = ctx.mode;
    const fragments = ctx.fragments;
    const quot = ctx.quot;
    const blocked = ctx.blocked;

    const suspects = makeUnique(ctx.suspects, 1);

    for (let n = 0; n < 42; n++) {
        // could be while(true), above is just a safety-net
        // in practice it will loop just a couple of times
        propagateModuleContextAndMethodName(suspects);
        if (!setChainedAndMethodNameThroughIifesAndReferences(suspects)) {
            break;
        }
    }

    // create final suspects by jumping, following, uniq'ing, blocking
    const finalSuspects = makeUnique(suspects.map(function(target) {
        const jumped = jumpOverIife(target);
        const jumpedAndFollowed = followReference(jumped) || jumped;

        if (target.$limitToMethodName && target.$limitToMethodName !== "*never*" && findOuterMethodName(target) !== target.$limitToMethodName) {
            return null;
        }

        if (blocked.indexOf(jumpedAndFollowed) >= 0) {
            return null;
        }

        return jumpedAndFollowed;
    }).filter(Boolean), 2);

    finalSuspects.forEach(function(target) {
        if (target.$chained !== chainedRegular) {
            return;
        }

        if (mode === "rebuild" && isAnnotatedArray(target)) {
            replaceArray(ctx, target, fragments, quot);
        } else if (mode === "remove" && isAnnotatedArray(target)) {
            removeArray(target, fragments);
        } else if (is.someof(mode, ["add", "rebuild"]) && isFunctionExpressionWithArgs(target)) {
            insertArray(ctx, target, fragments, quot);
        } else if (isGenericProviderName(target)) {
            renameProviderDeclarationSite(ctx, target, fragments);
        } else {
            // if it's not array or function-expression, then it's a candidate for foo.$inject = [..]
            judgeInjectArraySuspect(target, ctx);
        }
    });


    function propagateModuleContextAndMethodName(suspects) {
        suspects.forEach(function(target) {
            if (target.$chained !== chainedRegular && isInsideModuleContext(target)) {
                target.$chained = chainedRegular;
            }

            if (!target.$methodName) {
                const methodName = findOuterMethodName(target);
                if (methodName) {
                    target.$methodName = methodName;
                }
            }
        });
    }

    function findOuterMethodName(node) {
        for (; node && !node.$methodName; node = node.$parent) {
        }
        return node ? node.$methodName : null;
    }

    function setChainedAndMethodNameThroughIifesAndReferences(suspects) {
        let modified = false;
        suspects.forEach(function(target) {
            const jumped = jumpOverIife(target);
            if (jumped !== target) { // we did skip an IIFE
                if (target.$chained === chainedRegular && jumped.$chained !== chainedRegular) {
                    modified = true;
                    jumped.$chained = chainedRegular;
                }
                if (target.$methodName && !jumped.$methodName) {
                    modified = true;
                    jumped.$methodName = target.$methodName;
                }
            }

            const jumpedAndFollowed = followReference(jumped) || jumped;
            if (jumpedAndFollowed !== jumped) { // we did follow a reference
                if (jumped.$chained === chainedRegular && jumpedAndFollowed.$chained !== chainedRegular) {
                    modified = true;
                    jumpedAndFollowed.$chained = chainedRegular;
                }
                if (jumped.$methodName && !jumpedAndFollowed.$methodName) {
                    modified = true;
                    jumpedAndFollowed.$methodName = jumped.$methodName;
                }
            }
        });
        return modified;
    }

    function isInsideModuleContext(node) {
        let $parent = node.$parent;
        for (; $parent && $parent.$chained !== chainedRegular; $parent = $parent.$parent) {
        }
        return Boolean($parent);
    }

    function makeUnique(suspects, val) {
        return suspects.filter(function(target) {
            if (target.$seen === val) {
                return false;
            }
            target.$seen = val;
            return true;
        });
    }
}

function followReference(node) {
    if (!scopeTools.isReference(node)) {
        return null;
    }

    const scope = node.$scope.lookup(node.name);
    if (!scope) {
        return null;
    }

    const parent = scope.getNode(node.name).$parent;
    const kind = scope.getKind(node.name);
    const ptype = parent.type;

    if (is.someof(kind, ["const", "let", "var"])) {
        assert(ptype === "VariableDeclarator");
        // {type: "VariableDeclarator", id: {type: "Identifier", name: "foo"}, init: ..}
        return parent;
    } else if (kind === "fun") {
        assert(ptype === "FunctionDeclaration" || ptype === "FunctionExpression")
        // FunctionDeclaration is the common case, i.e.
        // function foo(a, b) {}

        // FunctionExpression is only applicable for cases similar to
        // var f = function asdf(a,b) { mymod.controller("asdf", asdf) };
        return parent;
    }

    // other kinds should not be handled ("param", "caught")

    return null;
}

// O(srclength) so should only be used for debugging purposes, else replace with lut
function posToLine(pos, src) {
    if (pos >= src.length) {
        pos = src.length - 1;
    }

    if (pos <= -1) {
        return -1;
    }

    let line = 1;
    for (let i = 0; i < pos; i++) {
        if (src[i] === "\n") {
            ++line;
        }
    }

    return line;
}

function judgeInjectArraySuspect(node, ctx) {
    if (node.type === "VariableDeclaration") {
        // suspect can only be a VariableDeclaration (statement) in case of
        // explicitly marked via /*@ngInject*/, not via references because
        // references follow to VariableDeclarator (child)

        // /*@ngInject*/ var foo = function($scope) {} and

        if (node.declarations.length !== 1) {
            // more than one declarator => exit
            return;
        }

        // one declarator => jump over declaration into declarator
        // rest of code will treat it as any (referenced) declarator
        node = node.declarations[0];
    }

    // onode is a top-level node (inside function block), later verified
    // node is inner match, descent in multiple steps
    let onode = null;
    let declaratorName = null;
    if (node.type === "VariableDeclarator") {
        onode = node.$parent;
        declaratorName = node.id.name;
        node = node.init; // var foo = ___;
    } else {
        onode = node;
    }

    // suspect must be inside of a block or at the top-level (i.e. inside of node.$parent.body[])
    if (!node || !onode.$parent || is.noneof(onode.$parent.type, ["Program", "BlockStatement"])) {
        return;
    }

    const insertPos = {
        pos: onode.range[1],
        loc: onode.loc.end
    };
    const isSemicolonTerminated = (ctx.src[insertPos.pos - 1] === ";");

    node = jumpOverIife(node);

    if (ctx.isFunctionExpressionWithArgs(node)) {
        // var x = 1, y = function(a,b) {}, z;

        assert(declaratorName);
        addRemoveInjectArray(
            node.params,
            isSemicolonTerminated ? insertPos : {
                pos: node.range[1],
                loc: node.loc.end
            },
            declaratorName);

    } else if (ctx.isFunctionDeclarationWithArgs(node)) {
        // /*@ngInject*/ function foo($scope) {}

        addRemoveInjectArray(
            node.params,
            insertPos,
            node.id.name);

    } else if (node.type === "ExpressionStatement" && node.expression.type === "AssignmentExpression" &&
        ctx.isFunctionExpressionWithArgs(node.expression.right)) {
        // /*@ngInject*/ foo.bar[0] = function($scope) {}

        const name = ctx.srcForRange(node.expression.left.range);
        addRemoveInjectArray(
            node.expression.right.params,
            isSemicolonTerminated ? insertPos : {
                pos: node.expression.right.range[1],
                loc: node.expression.right.loc.end
            },
            name);

    } else if (node = followReference(node)) {
        // node was a reference and followed node now is either a
        // FunctionDeclaration or a VariableDeclarator
        // => recurse

        judgeInjectArraySuspect(node, ctx);
    }


    function getIndent(pos) {
        const src = ctx.src;
        const lineStart = src.lastIndexOf("\n", pos - 1) + 1;
        let i = lineStart;
        for (; src[i] === " " || src[i] === "\t"; i++) {
        }
        return src.slice(lineStart, i);
    }

    function addRemoveInjectArray(params, posAfterFunctionDeclaration, name) {
        // if an existing something.$inject = [..] exists then is will always be recycled when rebuilding

        const indent = getIndent(posAfterFunctionDeclaration.pos);

        let foundSuspectInBody = false;
        let existingExpressionStatementWithArray = null;
        let troublesomeReturn = false;
        onode.$parent.body.forEach(function(bnode) {
            if (bnode === onode) {
                foundSuspectInBody = true;
            }

            if (hasInjectArray(bnode)) {
                if (existingExpressionStatementWithArray) {
                    throw fmt("conflicting inject arrays at line {0} and {1}",
                        posToLine(existingExpressionStatementWithArray.range[0], ctx.src),
                        posToLine(bnode.range[0], ctx.src));
                }
                existingExpressionStatementWithArray = bnode;
            }

            // there's a return statement before our function
            if (!foundSuspectInBody && bnode.type === "ReturnStatement") {
                troublesomeReturn = bnode;
            }
        });
        assert(foundSuspectInBody);

        if (troublesomeReturn && !existingExpressionStatementWithArray) {
            posAfterFunctionDeclaration = skipPrevNewline(troublesomeReturn.range[0], troublesomeReturn.loc.start);
        }

        function hasInjectArray(node) {
            let lvalue;
            let assignment;
            return (node && node.type === "ExpressionStatement" && (assignment = node.expression).type === "AssignmentExpression" &&
                assignment.operator === "=" &&
                (lvalue = assignment.left).type === "MemberExpression" &&
                ((lvalue.computed === false && ctx.srcForRange(lvalue.object.range) === name && lvalue.property.name === "$inject") ||
                    (lvalue.computed === true && ctx.srcForRange(lvalue.object.range) === name && lvalue.property.type === "Literal" && lvalue.property.value === "$inject")));
        }

        function skipPrevNewline(pos, loc) {
            let prevLF = ctx.src.lastIndexOf("\n", pos);
            if (prevLF === -1) {
                return { pos: pos, loc: loc };
            }
            if (prevLF >= 1 && ctx.src[prevLF] === "\r") {
                --prevLF;
            }

            if (/\S/g.test(ctx.src.slice(prevLF, pos - 1))) {
                return { pos: pos, loc: loc };
            }

            return {
                pos: prevLF,
                loc: {
                    line: loc.line - 1,
                    column: prevLF - ctx.src.lastIndexOf("\n", prevLF)
                }
            };
        }

        const str = fmt("{0}{1}{2}.$inject = {3};", EOL, indent, name, ctx.stringify(ctx, params, ctx.quot));

        if (ctx.mode === "rebuild" && existingExpressionStatementWithArray) {
            const strNoWhitespace = fmt("{2}.$inject = {3};", null, null, name, ctx.stringify(ctx, params, ctx.quot));
            ctx.fragments.push({
                start: existingExpressionStatementWithArray.range[0],
                end: existingExpressionStatementWithArray.range[1],
                str: strNoWhitespace,
                loc: {
                    start: existingExpressionStatementWithArray.loc.start,
                    end: existingExpressionStatementWithArray.loc.end
                }
            });
        } else if (ctx.mode === "remove" && existingExpressionStatementWithArray) {
            const start = skipPrevNewline(existingExpressionStatementWithArray.range[0], existingExpressionStatementWithArray.loc.start);
            ctx.fragments.push({
                start: start.pos,
                end: existingExpressionStatementWithArray.range[1],
                str: "",
                loc: {
                    start: start.loc,
                    end: existingExpressionStatementWithArray.loc.end
                }
            });
        } else if (is.someof(ctx.mode, ["add", "rebuild"]) && !existingExpressionStatementWithArray) {
            ctx.fragments.push({
                start: posAfterFunctionDeclaration.pos,
                end: posAfterFunctionDeclaration.pos,
                str: str,
                loc: {
                    start: posAfterFunctionDeclaration.loc,
                    end: posAfterFunctionDeclaration.loc
                }
            });
        }
    }
}

function jumpOverIife(node) {
    let outerfn;
    if (!(node.type === "CallExpression" && (outerfn = node.callee).type === "FunctionExpression")) {
        return node;
    }

    const outerbody = outerfn.body.body;
    for (let i = 0; i < outerbody.length; i++) {
        const statement = outerbody[i];
        if (statement.type === "ReturnStatement") {
            return statement.argument;
        }
    }

    return node;
}

function addModuleContextDependentSuspect(target, ctx) {
    ctx.suspects.push(target);
}

function addModuleContextIndependentSuspect(target, ctx) {
    target.$chained = chainedRegular;
    ctx.suspects.push(target);
}

function isAnnotatedArray(node) {
    if (node.type !== "ArrayExpression") {
        return false;
    }
    const elements = node.elements;

    // last should be a function expression
    if (elements.length === 0 || last(elements).type !== "FunctionExpression") {
        return false;
    }

    // all but last should be string literals
    for (let i = 0; i < elements.length - 1; i++) {
        const n = elements[i];
        if (n.type !== "Literal" || !is.string(n.value)) {
            return false;
        }
    }

    return true;
}
function isFunctionExpressionWithArgs(node) {
    return node.type === "FunctionExpression" && node.params.length >= 1;
}
function isFunctionDeclarationWithArgs(node) {
    return node.type === "FunctionDeclaration" && node.params.length >= 1;
}
function isGenericProviderName(node) {
    return node.type === "Literal" && is.string(node.value);
}

module.exports = function ngAnnotate(src, options) {
    const mode = (options.add && options.remove ? "rebuild" :
        options.remove ? "remove" :
            options.add ? "add" : null);

    if (!mode) {
        return {src: src};
    }

    const quot = options.single_quotes ? "'" : '"';
    const re = (options.regexp ? new RegExp(options.regexp) : /^[a-zA-Z0-9_\$\.\s]+$/);
    const rename = new stringmap();
    if (options.rename) {
        options.rename.forEach(function(value) {
            rename.set(value.from, value.to);
        });
    }
    let ast;
    const stats = {};

    // [{type: "Block"|"Line", value: str, range: [from,to]}, ..]
    let comments = [];

    try {
        stats.parser_require_t0 = require_acorn_t0;
        stats.parser_require_t1 = require_acorn_t1;
        stats.parser_parse_t0 = Date.now();
        // acorn
        ast = parser(src, {
            ecmaVersion: 6,
            locations: true,
            ranges: true,
            onComment: comments,
        });
        stats.parser_parse_t1 = Date.now();
    } catch(e) {
        return {
            errors: ["error: couldn't process source due to parse error", e.message],
        };
    }

    // append a dummy-node to ast so that lut.findNodeFromPos(lastPos) returns something
    ast.body.push({
        type: "DebuggerStatement",
        range: [ast.range[1], ast.range[1]],
        loc: {
            start: ast.loc.end,
            end: ast.loc.end
        }
    });

    // all source modifications are built up as operations in the
    // fragments array, later sent to alter in one shot
    const fragments = [];

    // suspects is built up with suspect nodes by match.
    // A suspect node will get annotations added / removed if it
    // fulfills the arrayexpression or functionexpression look,
    // and if it is in the correct context (inside an angular
    // module definition)
    const suspects = [];

    // blocked is an array of blocked suspects. Any target node
    // (final, i.e. IIFE-jumped, reference-followed and such) included
    // in blocked will be ignored by judgeSuspects
    const blocked = [];

    // Position information for all nodes in the AST,
    // used for sourcemap generation
    const nodePositions = [];

    const lut = new Lut(ast, src);

    scopeTools.setupScopeAndReferences(ast);

    const ctx = {
        mode: mode,
        quot: quot,
        src: src,
        srcForRange: function(range) {
            return src.slice(range[0], range[1]);
        },
        re: re,
        rename: rename,
        comments: comments,
        fragments: fragments,
        suspects: suspects,
        blocked: blocked,
        lut: lut,
        isFunctionExpressionWithArgs: isFunctionExpressionWithArgs,
        isFunctionDeclarationWithArgs: isFunctionDeclarationWithArgs,
        isAnnotatedArray: isAnnotatedArray,
        addModuleContextDependentSuspect: addModuleContextDependentSuspect,
        addModuleContextIndependentSuspect: addModuleContextIndependentSuspect,
        stringify: stringify,
        nodePositions: nodePositions,
    };

    const plugins = options.plugin || [];
    function matchPlugins(node, isMethodCall) {
        for (let i = 0; i < plugins.length; i++) {
            const res = plugins[i].match(node, isMethodCall);
            if (res) {
                return res;
            }
        }
        return false;
    }
    const matchPluginsOrNull = (plugins.length === 0 ? null : matchPlugins);

    ngInject.inspectComments(ctx);
    plugins.forEach(function(plugin) {
        plugin.init(ctx);
    });

    traverse(ast, {pre: function(node) {
        ngInject.inspectNode(node, ctx);

    }, post: function(node) {
        ctx.nodePositions.push(node.loc.start);
        let targets = match(node, ctx, matchPluginsOrNull);
        if (!targets) {
            return;
        }
        if (!is.array(targets)) {
            targets = [targets];
        }

        for (let i = 0; i < targets.length; i++) {
            addModuleContextDependentSuspect(targets[i], ctx);
        }
    }});

    try {
        judgeSuspects(ctx);
    } catch(e) {
        return {
            errors: ["error: " + e],
        };
    }

    const out = alter(src, fragments);
    const result = {
        src: out,
        _stats: stats,
    };

    if (options.sourcemap) {
        if (typeof(options.sourcemap) !== 'object')
            options.sourcemap = {};
        stats.sourcemap_t0 = Date.now();
        generateSourcemap(result, src, nodePositions, fragments, options.sourcemap);
        stats.sourcemap_t1 = Date.now();
    }

    return result;
}
