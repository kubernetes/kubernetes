/***********************************************************************

  A JavaScript tokenizer / parser / beautifier / compressor.
  https://github.com/mishoo/UglifyJS2

  -------------------------------- (C) ---------------------------------

                           Author: Mihai Bazon
                         <mihai.bazon@gmail.com>
                       http://mihai.bazon.net/blog

  Distributed under the BSD license:

    Copyright 2012 (c) Mihai Bazon <mihai.bazon@gmail.com>

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

        * Redistributions of source code must retain the above
          copyright notice, this list of conditions and the following
          disclaimer.

        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials
          provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER “AS IS” AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
    OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
    THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    SUCH DAMAGE.

 ***********************************************************************/

"use strict";

function find_builtins() {
    var a = [];
    [ Object, Array, Function, Number,
      String, Boolean, Error, Math,
      Date, RegExp
    ].forEach(function(ctor){
        Object.getOwnPropertyNames(ctor).map(add);
        if (ctor.prototype) {
            Object.getOwnPropertyNames(ctor.prototype).map(add);
        }
    });
    function add(name) {
        push_uniq(a, name);
    }
    return a;
}

function mangle_properties(ast, options) {
    options = defaults(options, {
        reserved : null,
        cache : null
    });

    var reserved = options.reserved;
    if (reserved == null)
        reserved = find_builtins();

    var cache = options.cache;
    if (cache == null) {
        cache = {
            cname: -1,
            props: new Dictionary()
        };
    }

    var names_to_mangle = [];

    // step 1: find candidates to mangle
    ast.walk(new TreeWalker(function(node){
        if (node instanceof AST_ObjectKeyVal) {
            add(node.key);
        }
        else if (node instanceof AST_ObjectProperty) {
            // setter or getter, since KeyVal is handled above
            add(node.key.name);
        }
        else if (node instanceof AST_Dot) {
            if (this.parent() instanceof AST_Assign) {
                add(node.property);
            }
        }
        else if (node instanceof AST_Sub) {
            if (this.parent() instanceof AST_Assign) {
                addStrings(node.property);
            }
        }
    }));

    // step 2: transform the tree, renaming properties
    return ast.transform(new TreeTransformer(null, function(node){
        if (node instanceof AST_ObjectKeyVal) {
            if (should_mangle(node.key)) {
                node.key = mangle(node.key);
            }
        }
        else if (node instanceof AST_ObjectProperty) {
            // setter or getter
            if (should_mangle(node.key.name)) {
                node.key.name = mangle(node.key.name);
            }
        }
        else if (node instanceof AST_Dot) {
            if (should_mangle(node.property)) {
                node.property = mangle(node.property);
            }
        }
        else if (node instanceof AST_Sub) {
            node.property = mangleStrings(node.property);
        }
        // else if (node instanceof AST_String) {
        //     if (should_mangle(node.value)) {
        //         AST_Node.warn(
        //             "Found \"{prop}\" property candidate for mangling in an arbitrary string [{file}:{line},{col}]", {
        //                 file : node.start.file,
        //                 line : node.start.line,
        //                 col  : node.start.col,
        //                 prop : node.value
        //             }
        //         );
        //     }
        // }
    }));

    // only function declarations after this line

    function can_mangle(name) {
        if (reserved.indexOf(name) >= 0) return false;
        if (/^[0-9.]+$/.test(name)) return false;
        return true;
    }

    function should_mangle(name) {
        return cache.props.has(name)
            || names_to_mangle.indexOf(name) >= 0;
    }

    function add(name) {
        if (can_mangle(name))
            push_uniq(names_to_mangle, name);
    }

    function mangle(name) {
        var mangled = cache.props.get(name);
        if (!mangled) {
            do {
                mangled = base54(++cache.cname);
            } while (!can_mangle(mangled));
            cache.props.set(name, mangled);
        }
        return mangled;
    }

    function addStrings(node) {
        var out = {};
        try {
            (function walk(node){
                node.walk(new TreeWalker(function(node){
                    if (node instanceof AST_Seq) {
                        walk(node.cdr);
                        return true;
                    }
                    if (node instanceof AST_String) {
                        add(node.value);
                        return true;
                    }
                    if (node instanceof AST_Conditional) {
                        walk(node.consequent);
                        walk(node.alternative);
                        return true;
                    }
                    throw out;
                }));
            })(node);
        } catch(ex) {
            if (ex !== out) throw ex;
        }
    }

    function mangleStrings(node) {
        return node.transform(new TreeTransformer(function(node){
            if (node instanceof AST_Seq) {
                node.cdr = mangleStrings(node.cdr);
            }
            else if (node instanceof AST_String) {
                if (should_mangle(node.value)) {
                    node.value = mangle(node.value);
                }
            }
            else if (node instanceof AST_Conditional) {
                node.consequent = mangleStrings(node.consequent);
                node.alternative = mangleStrings(node.alternative);
            }
            return node;
        }));
    }

}
