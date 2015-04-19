(function (tree) {
    tree.toCSSVisitor = function(env) {
        this._visitor = new tree.visitor(this);
        this._env = env;
    };

    tree.toCSSVisitor.prototype = {
        isReplacing: true,
        run: function (root) {
            return this._visitor.visit(root);
        },

        visitRule: function (ruleNode, visitArgs) {
            if (ruleNode.variable) {
                return [];
            }
            return ruleNode;
        },

        visitMixinDefinition: function (mixinNode, visitArgs) {
            // mixin definitions do not get eval'd - this means they keep state
            // so we have to clear that state here so it isn't used if toCSS is called twice
            mixinNode.frames = [];
            return [];
        },

        visitExtend: function (extendNode, visitArgs) {
            return [];
        },

        visitComment: function (commentNode, visitArgs) {
            if (commentNode.isSilent(this._env)) {
                return [];
            }
            return commentNode;
        },

        visitMedia: function(mediaNode, visitArgs) {
            mediaNode.accept(this._visitor);
            visitArgs.visitDeeper = false;

            if (!mediaNode.rules.length) {
                return [];
            }
            return mediaNode;
        },

        visitDirective: function(directiveNode, visitArgs) {
            if (directiveNode.currentFileInfo.reference && !directiveNode.isReferenced) {
                return [];
            }
            if (directiveNode.name === "@charset") {
                // Only output the debug info together with subsequent @charset definitions
                // a comment (or @media statement) before the actual @charset directive would
                // be considered illegal css as it has to be on the first line
                if (this.charset) {
                    if (directiveNode.debugInfo) {
                        var comment = new tree.Comment("/* " + directiveNode.toCSS(this._env).replace(/\n/g, "")+" */\n");
                        comment.debugInfo = directiveNode.debugInfo;
                        return this._visitor.visit(comment);
                    }
                    return [];
                }
                this.charset = true;
            }
            if (directiveNode.rules && directiveNode.rules.rules) {
                this._mergeRules(directiveNode.rules.rules);
            }
            return directiveNode;
        },

        checkPropertiesInRoot: function(rules) {
            var ruleNode;
            for(var i = 0; i < rules.length; i++) {
                ruleNode = rules[i];
                if (ruleNode instanceof tree.Rule && !ruleNode.variable) {
                    throw { message: "properties must be inside selector blocks, they cannot be in the root.",
                        index: ruleNode.index, filename: ruleNode.currentFileInfo ? ruleNode.currentFileInfo.filename : null};
                }
            }
        },

        visitRuleset: function (rulesetNode, visitArgs) {
            var rule, rulesets = [];
            if (rulesetNode.firstRoot) {
                this.checkPropertiesInRoot(rulesetNode.rules);
            }
            if (! rulesetNode.root) {
                if (rulesetNode.paths) {
                    rulesetNode.paths = rulesetNode.paths
                        .filter(function(p) {
                            var i;
                            if (p[0].elements[0].combinator.value === ' ') {
                                p[0].elements[0].combinator = new(tree.Combinator)('');
                            }
                            for(i = 0; i < p.length; i++) {
                                if (p[i].getIsReferenced() && p[i].getIsOutput()) {
                                    return true;
                                }
                            }
                            return false;
                        });
                }

                // Compile rules and rulesets
                var nodeRules = rulesetNode.rules, nodeRuleCnt = nodeRules ? nodeRules.length : 0;
                for (var i = 0; i < nodeRuleCnt; ) {
                    rule = nodeRules[i];
                    if (rule && rule.rules) {
                        // visit because we are moving them out from being a child
                        rulesets.push(this._visitor.visit(rule));
                        nodeRules.splice(i, 1);
                        nodeRuleCnt--;
                        continue;
                    }
                    i++;
                }
                // accept the visitor to remove rules and refactor itself
                // then we can decide now whether we want it or not
                if (nodeRuleCnt > 0) {
                    rulesetNode.accept(this._visitor);
                } else {
                    rulesetNode.rules = null;
                }
                visitArgs.visitDeeper = false;

                nodeRules = rulesetNode.rules;
                if (nodeRules) {
                    this._mergeRules(nodeRules);
                    nodeRules = rulesetNode.rules;
                }
                if (nodeRules) {
                    this._removeDuplicateRules(nodeRules);
                    nodeRules = rulesetNode.rules;
                }

                // now decide whether we keep the ruleset
                if (nodeRules && nodeRules.length > 0 && rulesetNode.paths.length > 0) {
                    rulesets.splice(0, 0, rulesetNode);
                }
            } else {
                rulesetNode.accept(this._visitor);
                visitArgs.visitDeeper = false;
                if (rulesetNode.firstRoot || (rulesetNode.rules && rulesetNode.rules.length > 0)) {
                    rulesets.splice(0, 0, rulesetNode);
                }
            }
            if (rulesets.length === 1) {
                return rulesets[0];
            }
            return rulesets;
        },

        _removeDuplicateRules: function(rules) {
            if (!rules) { return; }

            // remove duplicates
            var ruleCache = {},
                ruleList, rule, i;

            for(i = rules.length - 1; i >= 0 ; i--) {
                rule = rules[i];
                if (rule instanceof tree.Rule) {
                    if (!ruleCache[rule.name]) {
                        ruleCache[rule.name] = rule;
                    } else {
                        ruleList = ruleCache[rule.name];
                        if (ruleList instanceof tree.Rule) {
                            ruleList = ruleCache[rule.name] = [ruleCache[rule.name].toCSS(this._env)];
                        }
                        var ruleCSS = rule.toCSS(this._env);
                        if (ruleList.indexOf(ruleCSS) !== -1) {
                            rules.splice(i, 1);
                        } else {
                            ruleList.push(ruleCSS);
                        }
                    }
                }
            }
        },

        _mergeRules: function (rules) {
            if (!rules) { return; }

            var groups = {},
                parts,
                rule,
                key;

            for (var i = 0; i < rules.length; i++) {
                rule = rules[i];

                if ((rule instanceof tree.Rule) && rule.merge) {
                    key = [rule.name,
                        rule.important ? "!" : ""].join(",");

                    if (!groups[key]) {
                        groups[key] = [];
                    } else {
                        rules.splice(i--, 1);
                    }

                    groups[key].push(rule);
                }
            }

            Object.keys(groups).map(function (k) {

                function toExpression(values) {
                    return new (tree.Expression)(values.map(function (p) {
                        return p.value;
                    }));
                }

                function toValue(values) {
                    return new (tree.Value)(values.map(function (p) {
                        return p;
                    }));
                }

                parts = groups[k];

                if (parts.length > 1) {
                    rule = parts[0];
                    var spacedGroups = [];
                    var lastSpacedGroup = [];
                    parts.map(function (p) {
                    if (p.merge==="+") {
                        if (lastSpacedGroup.length > 0) {
                                spacedGroups.push(toExpression(lastSpacedGroup));
                            }
                            lastSpacedGroup = [];
                        }
                        lastSpacedGroup.push(p);
                    });
                    spacedGroups.push(toExpression(lastSpacedGroup));
                    rule.value = toValue(spacedGroups);
                }
            });
        }
    };

})(require('./tree'));