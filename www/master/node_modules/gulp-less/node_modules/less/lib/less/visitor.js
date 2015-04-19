(function (tree) {

    var _visitArgs = { visitDeeper: true },
        _hasIndexed = false;

    function _noop(node) {
        return node;
    }

    function indexNodeTypes(parent, ticker) {
        // add .typeIndex to tree node types for lookup table
        var key, child;
        for (key in parent) {
            if (parent.hasOwnProperty(key)) {
                child = parent[key];
                switch (typeof child) {
                    case "function":
                        // ignore bound functions directly on tree which do not have a prototype
                        // or aren't nodes
                        if (child.prototype && child.prototype.type) {
                            child.prototype.typeIndex = ticker++;
                        }
                        break;
                    case "object":
                        ticker = indexNodeTypes(child, ticker);
                        break;
                }
            }
        }
        return ticker;
    }

    tree.visitor = function(implementation) {
        this._implementation = implementation;
        this._visitFnCache = [];

        if (!_hasIndexed) {
            indexNodeTypes(tree, 1);
            _hasIndexed = true;
        }
    };

    tree.visitor.prototype = {
        visit: function(node) {
            if (!node) {
                return node;
            }

            var nodeTypeIndex = node.typeIndex;
            if (!nodeTypeIndex) {
                return node;
            }

            var visitFnCache = this._visitFnCache,
                impl = this._implementation,
                aryIndx = nodeTypeIndex << 1,
                outAryIndex = aryIndx | 1,
                func = visitFnCache[aryIndx],
                funcOut = visitFnCache[outAryIndex],
                visitArgs = _visitArgs,
                fnName;

            visitArgs.visitDeeper = true;

            if (!func) {
                fnName = "visit" + node.type;
                func = impl[fnName] || _noop;
                funcOut = impl[fnName + "Out"] || _noop;
                visitFnCache[aryIndx] = func;
                visitFnCache[outAryIndex] = funcOut;
            }

            if (func !== _noop) {
                var newNode = func.call(impl, node, visitArgs);
                if (impl.isReplacing) {
                    node = newNode;
                }
            }

            if (visitArgs.visitDeeper && node && node.accept) {
                node.accept(this);
            }

            if (funcOut != _noop) {
                funcOut.call(impl, node);
            }

            return node;
        },
        visitArray: function(nodes, nonReplacing) {
            if (!nodes) {
                return nodes;
            }

            var cnt = nodes.length, i;

            // Non-replacing
            if (nonReplacing || !this._implementation.isReplacing) {
                for (i = 0; i < cnt; i++) {
                    this.visit(nodes[i]);
                }
                return nodes;
            }

            // Replacing
            var out = [];
            for (i = 0; i < cnt; i++) {
                var evald = this.visit(nodes[i]);
                if (!evald.splice) {
                    out.push(evald);
                } else if (evald.length) {
                    this.flatten(evald, out);
                }
            }
            return out;
        },
        flatten: function(arr, out) {
            if (!out) {
                out = [];
            }

            var cnt, i, item,
                nestedCnt, j, nestedItem;

            for (i = 0, cnt = arr.length; i < cnt; i++) {
                item = arr[i];
                if (!item.splice) {
                    out.push(item);
                    continue;
                }

                for (j = 0, nestedCnt = item.length; j < nestedCnt; j++) {
                    nestedItem = item[j];
                    if (!nestedItem.splice) {
                        out.push(nestedItem);
                    } else if (nestedItem.length) {
                        this.flatten(nestedItem, out);
                    }
                }
            }

            return out;
        }
    };

})(require('./tree'));