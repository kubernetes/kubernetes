// ordered-ast-traverse.js
// MIT licensed, see LICENSE file
// Copyright (c) 2014-2015 Olov Lassus <olov.lassus@gmail.com>

"use strict";

var props = require("ordered-esprima-props");
var noProps = [];

function traverse(root, options) {
    "use strict";

    options = options || {};
    var pre = options.pre;
    var post = options.post;
    var skipProperty = options.skipProperty;

    function visit(node, parent, prop, idx) {
        if (!node || typeof node.type !== "string") {
            return;
        }

        var res = undefined;
        if (pre) {
            res = pre(node, parent, prop, idx);
        }

        if (res !== false) {
            var nodeProps = (props[node.type] || noProps);

            for (var idx = 0; idx < nodeProps.length; idx++) {
                var prop = nodeProps[idx];

                if (skipProperty && skipProperty(prop, node)) {
                    continue;
                }

                var child = node[prop];

                if (Array.isArray(child)) {
                    for (var i = 0; i < child.length; i++) {
                        var c = child[i];
                        visit(c, node, prop, i);
                    }
                } else {
                    visit(child, node, prop);
                }
            }
        }

        if (post) {
            post(node, parent, prop, idx);
        }
    }

    visit(root, null);
};

if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
    module.exports = traverse;
}
