(function (tree) {
    tree.joinSelectorVisitor = function() {
        this.contexts = [[]];
        this._visitor = new tree.visitor(this);
    };

    tree.joinSelectorVisitor.prototype = {
        run: function (root) {
            return this._visitor.visit(root);
        },
        visitRule: function (ruleNode, visitArgs) {
            visitArgs.visitDeeper = false;
        },
        visitMixinDefinition: function (mixinDefinitionNode, visitArgs) {
            visitArgs.visitDeeper = false;
        },

        visitRuleset: function (rulesetNode, visitArgs) {
            var context = this.contexts[this.contexts.length - 1],
                paths = [], selectors;

            this.contexts.push(paths);

            if (! rulesetNode.root) {
                selectors = rulesetNode.selectors;
                if (selectors) {
                    selectors = selectors.filter(function(selector) { return selector.getIsOutput(); });
                    rulesetNode.selectors = selectors.length ? selectors : (selectors = null);
                    if (selectors) { rulesetNode.joinSelectors(paths, context, selectors); }
                }
                if (!selectors) { rulesetNode.rules = null; }
                rulesetNode.paths = paths;
            }
        },
        visitRulesetOut: function (rulesetNode) {
            this.contexts.length = this.contexts.length - 1;
        },
        visitMedia: function (mediaNode, visitArgs) {
            var context = this.contexts[this.contexts.length - 1];
            mediaNode.rules[0].root = (context.length === 0 || context[0].multiMedia);
        }
    };

})(require('./tree'));