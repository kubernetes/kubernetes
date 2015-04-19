var jsp = require("./parse-js"),
    pro = require("./process");

var BY_TYPE = {};

function HOP(obj, prop) {
        return Object.prototype.hasOwnProperty.call(obj, prop);
};

function AST_Node(parent) {
        this.parent = parent;
};

AST_Node.prototype.init = function(){};

function DEFINE_NODE_CLASS(type, props, methods) {
        var base = methods && methods.BASE || AST_Node;
        if (!base) base = AST_Node;
        function D(parent, data) {
                base.apply(this, arguments);
                if (props) props.forEach(function(name, i){
                        this["_" + name] = data[i];
                });
                this.init();
        };
        var P = D.prototype = new AST_Node;
        P.node_type = function(){ return type };
        if (props) props.forEach(function(name){
                var propname = "_" + name;
                P["set_" + name] = function(val) {
                        this[propname] = val;
                        return this;
                };
                P["get_" + name] = function() {
                        return this[propname];
                };
        });
        if (type != null) BY_TYPE[type] = D;
        if (methods) for (var i in methods) if (HOP(methods, i)) {
                P[i] = methods[i];
        }
        return D;
};

var AST_String_Node = DEFINE_NODE_CLASS("string", ["value"]);
var AST_Number_Node = DEFINE_NODE_CLASS("num", ["value"]);
var AST_Name_Node = DEFINE_NODE_CLASS("name", ["value"]);

var AST_Statlist_Node = DEFINE_NODE_CLASS(null, ["body"]);
var AST_Root_Node = DEFINE_NODE_CLASS("toplevel", null, { BASE: AST_Statlist_Node });
var AST_Block_Node = DEFINE_NODE_CLASS("block", null, { BASE: AST_Statlist_Node });
var AST_Splice_Node = DEFINE_NODE_CLASS("splice", null, { BASE: AST_Statlist_Node });

var AST_Var_Node = DEFINE_NODE_CLASS("var", ["definitions"]);
var AST_Const_Node = DEFINE_NODE_CLASS("const", ["definitions"]);

var AST_Try_Node = DEFINE_NODE_CLASS("try", ["body", "catch", "finally"]);
var AST_Throw_Node = DEFINE_NODE_CLASS("throw", ["exception"]);

var AST_New_Node = DEFINE_NODE_CLASS("new", ["constructor", "arguments"]);

var AST_Switch_Node = DEFINE_NODE_CLASS("switch", ["expression", "branches"]);
var AST_Switch_Branch_Node = DEFINE_NODE_CLASS(null, ["expression", "body"]);

var AST_Break_Node = DEFINE_NODE_CLASS("break", ["label"]);
var AST_Continue_Node = DEFINE_NODE_CLASS("continue", ["label"]);
var AST_Assign_Node = DEFINE_NODE_CLASS("assign", ["operator", "lvalue", "rvalue"]);
var AST_Dot_Node = DEFINE_NODE_CLASS("dot", ["expression", "name"]);
var AST_Call_Node = DEFINE_NODE_CLASS("call", ["function", "arguments"]);

var AST_Lambda_Node = DEFINE_NODE_CLASS(null, ["name", "arguments", "body"])
var AST_Function_Node = DEFINE_NODE_CLASS("function", null, AST_Lambda_Node);
var AST_Defun_Node = DEFINE_NODE_CLASS("defun", null, AST_Lambda_Node);

var AST_If_Node = DEFINE_NODE_CLASS("if", ["condition", "then", "else"]);
