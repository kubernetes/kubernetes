(function (tree) {

tree.Ruleset = function (selectors, rules, strictImports) {
    this.selectors = selectors;
    this.rules = rules;
    this._lookups = {};
    this.strictImports = strictImports;
};
tree.Ruleset.prototype = {
    type: "Ruleset",
    accept: function (visitor) {
        if (this.paths) {
            visitor.visitArray(this.paths, true);
        } else if (this.selectors) {
            this.selectors = visitor.visitArray(this.selectors);
        }
        if (this.rules && this.rules.length) {
            this.rules = visitor.visitArray(this.rules);
        }
    },
    eval: function (env) {
        var thisSelectors = this.selectors, selectors, 
            selCnt, selector, i, defaultFunc = tree.defaultFunc, hasOnePassingSelector = false;

        if (thisSelectors && (selCnt = thisSelectors.length)) {
            selectors = [];
            defaultFunc.error({
                type: "Syntax", 
                message: "it is currently only allowed in parametric mixin guards," 
            });
            for (i = 0; i < selCnt; i++) {
                selector = thisSelectors[i].eval(env);
                selectors.push(selector);
                if (selector.evaldCondition) {
                    hasOnePassingSelector = true;
                }
            }
            defaultFunc.reset();  
        } else {
            hasOnePassingSelector = true;
        }

        var rules = this.rules ? this.rules.slice(0) : null,
            ruleset = new(tree.Ruleset)(selectors, rules, this.strictImports),
            rule, subRule;

        ruleset.originalRuleset = this;
        ruleset.root = this.root;
        ruleset.firstRoot = this.firstRoot;
        ruleset.allowImports = this.allowImports;

        if(this.debugInfo) {
            ruleset.debugInfo = this.debugInfo;
        }
        
        if (!hasOnePassingSelector) {
            rules.length = 0;
        }

        // push the current ruleset to the frames stack
        var envFrames = env.frames;
        envFrames.unshift(ruleset);

        // currrent selectors
        var envSelectors = env.selectors;
        if (!envSelectors) {
            env.selectors = envSelectors = [];
        }
        envSelectors.unshift(this.selectors);

        // Evaluate imports
        if (ruleset.root || ruleset.allowImports || !ruleset.strictImports) {
            ruleset.evalImports(env);
        }

        // Store the frames around mixin definitions,
        // so they can be evaluated like closures when the time comes.
        var rsRules = ruleset.rules, rsRuleCnt = rsRules ? rsRules.length : 0;
        for (i = 0; i < rsRuleCnt; i++) {
            if (rsRules[i] instanceof tree.mixin.Definition || rsRules[i] instanceof tree.DetachedRuleset) {
                rsRules[i] = rsRules[i].eval(env);
            }
        }

        var mediaBlockCount = (env.mediaBlocks && env.mediaBlocks.length) || 0;

        // Evaluate mixin calls.
        for (i = 0; i < rsRuleCnt; i++) {
            if (rsRules[i] instanceof tree.mixin.Call) {
                /*jshint loopfunc:true */
                rules = rsRules[i].eval(env).filter(function(r) {
                    if ((r instanceof tree.Rule) && r.variable) {
                        // do not pollute the scope if the variable is
                        // already there. consider returning false here
                        // but we need a way to "return" variable from mixins
                        return !(ruleset.variable(r.name));
                    }
                    return true;
                });
                rsRules.splice.apply(rsRules, [i, 1].concat(rules));
                rsRuleCnt += rules.length - 1;
                i += rules.length-1;
                ruleset.resetCache();
            } else if (rsRules[i] instanceof tree.RulesetCall) {
                /*jshint loopfunc:true */
                rules = rsRules[i].eval(env).rules.filter(function(r) {
                    if ((r instanceof tree.Rule) && r.variable) {
                        // do not pollute the scope at all
                        return false;
                    }
                    return true;
                });
                rsRules.splice.apply(rsRules, [i, 1].concat(rules));
                rsRuleCnt += rules.length - 1;
                i += rules.length-1;
                ruleset.resetCache();
            }
        }

        // Evaluate everything else
        for (i = 0; i < rsRules.length; i++) {
            rule = rsRules[i];
            if (! (rule instanceof tree.mixin.Definition || rule instanceof tree.DetachedRuleset)) {
                rsRules[i] = rule = rule.eval ? rule.eval(env) : rule;
            }
        }
        
        // Evaluate everything else
        for (i = 0; i < rsRules.length; i++) {
            rule = rsRules[i];
            // for rulesets, check if it is a css guard and can be removed
            if (rule instanceof tree.Ruleset && rule.selectors && rule.selectors.length === 1) {
                // check if it can be folded in (e.g. & where)
                if (rule.selectors[0].isJustParentSelector()) {
                    rsRules.splice(i--, 1);

                    for(var j = 0; j < rule.rules.length; j++) {
                        subRule = rule.rules[j];
                        if (!(subRule instanceof tree.Rule) || !subRule.variable) {
                            rsRules.splice(++i, 0, subRule);
                        }
                    }
                }
            }
        }

        // Pop the stack
        envFrames.shift();
        envSelectors.shift();
        
        if (env.mediaBlocks) {
            for (i = mediaBlockCount; i < env.mediaBlocks.length; i++) {
                env.mediaBlocks[i].bubbleSelectors(selectors);
            }
        }

        return ruleset;
    },
    evalImports: function(env) {
        var rules = this.rules, i, importRules;
        if (!rules) { return; }

        for (i = 0; i < rules.length; i++) {
            if (rules[i] instanceof tree.Import) {
                importRules = rules[i].eval(env);
                if (importRules && importRules.length) {
                    rules.splice.apply(rules, [i, 1].concat(importRules));
                    i+= importRules.length-1;
                } else {
                    rules.splice(i, 1, importRules);
                }
                this.resetCache();
            }
        }
    },
    makeImportant: function() {
        return new tree.Ruleset(this.selectors, this.rules.map(function (r) {
                    if (r.makeImportant) {
                        return r.makeImportant();
                    } else {
                        return r;
                    }
                }), this.strictImports);
    },
    matchArgs: function (args) {
        return !args || args.length === 0;
    },
    // lets you call a css selector with a guard
    matchCondition: function (args, env) {
        var lastSelector = this.selectors[this.selectors.length-1];
        if (!lastSelector.evaldCondition) {
            return false;
        }
        if (lastSelector.condition &&
            !lastSelector.condition.eval(
                new(tree.evalEnv)(env,
                    env.frames))) {
            return false;
        }
        return true;
    },
    resetCache: function () {
        this._rulesets = null;
        this._variables = null;
        this._lookups = {};
    },
    variables: function () {
        if (!this._variables) {
            this._variables = !this.rules ? {} : this.rules.reduce(function (hash, r) {
                if (r instanceof tree.Rule && r.variable === true) {
                    hash[r.name] = r;
                }
                return hash;
            }, {});
        }
        return this._variables;
    },
    variable: function (name) {
        return this.variables()[name];
    },
    rulesets: function () {
        if (!this.rules) { return null; }

        var _Ruleset = tree.Ruleset, _MixinDefinition = tree.mixin.Definition,
            filtRules = [], rules = this.rules, cnt = rules.length,
            i, rule;

        for (i = 0; i < cnt; i++) {
            rule = rules[i];
            if ((rule instanceof _Ruleset) || (rule instanceof _MixinDefinition)) {
                filtRules.push(rule);
            }
        }

        return filtRules;
    },
    prependRule: function (rule) {
        var rules = this.rules;
        if (rules) { rules.unshift(rule); } else { this.rules = [ rule ]; }
    },
    find: function (selector, self) {
        self = self || this;
        var rules = [], match,
            key = selector.toCSS();

        if (key in this._lookups) { return this._lookups[key]; }

        this.rulesets().forEach(function (rule) {
            if (rule !== self) {
                for (var j = 0; j < rule.selectors.length; j++) {
                    match = selector.match(rule.selectors[j]);
                    if (match) {
                        if (selector.elements.length > match) {
                            Array.prototype.push.apply(rules, rule.find(
                                new(tree.Selector)(selector.elements.slice(match)), self));
                        } else {
                            rules.push(rule);
                        }
                        break;
                    }
                }
            }
        });
        this._lookups[key] = rules;
        return rules;
    },
    genCSS: function (env, output) {
        var i, j,
            charsetRuleNodes = [],
            ruleNodes = [],
            rulesetNodes = [],
            rulesetNodeCnt,
            debugInfo,     // Line number debugging
            rule,
            path;

        env.tabLevel = (env.tabLevel || 0);

        if (!this.root) {
            env.tabLevel++;
        }

        var tabRuleStr = env.compress ? '' : Array(env.tabLevel + 1).join("  "),
            tabSetStr = env.compress ? '' : Array(env.tabLevel).join("  "),
            sep;

        function isRulesetLikeNode(rule, root) {
             // if it has nested rules, then it should be treated like a ruleset
             if (rule.rules)
                 return true;

             // medias and comments do not have nested rules, but should be treated like rulesets anyway
             if ( (rule instanceof tree.Media) || (root && rule instanceof tree.Comment))
                 return true;

             // some directives and anonumoust nodes are ruleset like, others are not
             if ((rule instanceof tree.Directive) || (rule instanceof tree.Anonymous)) {
                 return rule.isRulesetLike();
             }

             //anything else is assumed to be a rule
             return false;
        }

        for (i = 0; i < this.rules.length; i++) {
            rule = this.rules[i];
            if (isRulesetLikeNode(rule, this.root)) {
                rulesetNodes.push(rule);
            } else {
                //charsets should float on top of everything
                if (rule.isCharset && rule.isCharset()) {
                    charsetRuleNodes.push(rule);
                } else {
                    ruleNodes.push(rule);
                }
            }
        }
        ruleNodes = charsetRuleNodes.concat(ruleNodes);

        // If this is the root node, we don't render
        // a selector, or {}.
        if (!this.root) {
            debugInfo = tree.debugInfo(env, this, tabSetStr);

            if (debugInfo) {
                output.add(debugInfo);
                output.add(tabSetStr);
            }

            var paths = this.paths, pathCnt = paths.length,
                pathSubCnt;

            sep = env.compress ? ',' : (',\n' + tabSetStr);

            for (i = 0; i < pathCnt; i++) {
                path = paths[i];
                if (!(pathSubCnt = path.length)) { continue; }
                if (i > 0) { output.add(sep); }

                env.firstSelector = true;
                path[0].genCSS(env, output);

                env.firstSelector = false;
                for (j = 1; j < pathSubCnt; j++) {
                    path[j].genCSS(env, output);
                }
            }

            output.add((env.compress ? '{' : ' {\n') + tabRuleStr);
        }

        // Compile rules and rulesets
        for (i = 0; i < ruleNodes.length; i++) {
            rule = ruleNodes[i];

            // @page{ directive ends up with root elements inside it, a mix of rules and rulesets
            // In this instance we do not know whether it is the last property
            if (i + 1 === ruleNodes.length && (!this.root || rulesetNodes.length === 0 || this.firstRoot)) {
                env.lastRule = true;
            }

            if (rule.genCSS) {
                rule.genCSS(env, output);
            } else if (rule.value) {
                output.add(rule.value.toString());
            }

            if (!env.lastRule) {
                output.add(env.compress ? '' : ('\n' + tabRuleStr));
            } else {
                env.lastRule = false;
            }
        }

        if (!this.root) {
            output.add((env.compress ? '}' : '\n' + tabSetStr + '}'));
            env.tabLevel--;
        }

        sep = (env.compress ? "" : "\n") + (this.root ? tabRuleStr : tabSetStr);
        rulesetNodeCnt = rulesetNodes.length;
        if (rulesetNodeCnt) {
            if (ruleNodes.length && sep) { output.add(sep); }
            rulesetNodes[0].genCSS(env, output);
            for (i = 1; i < rulesetNodeCnt; i++) {
                if (sep) { output.add(sep); }
                rulesetNodes[i].genCSS(env, output);
            }
        }

        if (!output.isEmpty() && !env.compress && this.firstRoot) {
            output.add('\n');
        }
    },

    toCSS: tree.toCSS,

    markReferenced: function () {
        if (!this.selectors) {
            return;
        }
        for (var s = 0; s < this.selectors.length; s++) {
            this.selectors[s].markReferenced();
        }
    },

    joinSelectors: function (paths, context, selectors) {
        for (var s = 0; s < selectors.length; s++) {
            this.joinSelector(paths, context, selectors[s]);
        }
    },

    joinSelector: function (paths, context, selector) {

        var i, j, k, 
            hasParentSelector, newSelectors, el, sel, parentSel, 
            newSelectorPath, afterParentJoin, newJoinedSelector, 
            newJoinedSelectorEmpty, lastSelector, currentElements,
            selectorsMultiplied;
    
        for (i = 0; i < selector.elements.length; i++) {
            el = selector.elements[i];
            if (el.value === '&') {
                hasParentSelector = true;
            }
        }
    
        if (!hasParentSelector) {
            if (context.length > 0) {
                for (i = 0; i < context.length; i++) {
                    paths.push(context[i].concat(selector));
                }
            }
            else {
                paths.push([selector]);
            }
            return;
        }

        // The paths are [[Selector]]
        // The first list is a list of comma seperated selectors
        // The inner list is a list of inheritance seperated selectors
        // e.g.
        // .a, .b {
        //   .c {
        //   }
        // }
        // == [[.a] [.c]] [[.b] [.c]]
        //

        // the elements from the current selector so far
        currentElements = [];
        // the current list of new selectors to add to the path.
        // We will build it up. We initiate it with one empty selector as we "multiply" the new selectors
        // by the parents
        newSelectors = [[]];

        for (i = 0; i < selector.elements.length; i++) {
            el = selector.elements[i];
            // non parent reference elements just get added
            if (el.value !== "&") {
                currentElements.push(el);
            } else {
                // the new list of selectors to add
                selectorsMultiplied = [];

                // merge the current list of non parent selector elements
                // on to the current list of selectors to add
                if (currentElements.length > 0) {
                    this.mergeElementsOnToSelectors(currentElements, newSelectors);
                }

                // loop through our current selectors
                for (j = 0; j < newSelectors.length; j++) {
                    sel = newSelectors[j];
                    // if we don't have any parent paths, the & might be in a mixin so that it can be used
                    // whether there are parents or not
                    if (context.length === 0) {
                        // the combinator used on el should now be applied to the next element instead so that
                        // it is not lost
                        if (sel.length > 0) {
                            sel[0].elements = sel[0].elements.slice(0);
                            sel[0].elements.push(new(tree.Element)(el.combinator, '', el.index, el.currentFileInfo));
                        }
                        selectorsMultiplied.push(sel);
                    }
                    else {
                        // and the parent selectors
                        for (k = 0; k < context.length; k++) {
                            parentSel = context[k];
                            // We need to put the current selectors
                            // then join the last selector's elements on to the parents selectors

                            // our new selector path
                            newSelectorPath = [];
                            // selectors from the parent after the join
                            afterParentJoin = [];
                            newJoinedSelectorEmpty = true;

                            //construct the joined selector - if & is the first thing this will be empty,
                            // if not newJoinedSelector will be the last set of elements in the selector
                            if (sel.length > 0) {
                                newSelectorPath = sel.slice(0);
                                lastSelector = newSelectorPath.pop();
                                newJoinedSelector = selector.createDerived(lastSelector.elements.slice(0));
                                newJoinedSelectorEmpty = false;
                            }
                            else {
                                newJoinedSelector = selector.createDerived([]);
                            }

                            //put together the parent selectors after the join
                            if (parentSel.length > 1) {
                                afterParentJoin = afterParentJoin.concat(parentSel.slice(1));
                            }

                            if (parentSel.length > 0) {
                                newJoinedSelectorEmpty = false;

                                // join the elements so far with the first part of the parent
                                newJoinedSelector.elements.push(new(tree.Element)(el.combinator, parentSel[0].elements[0].value, el.index, el.currentFileInfo));
                                newJoinedSelector.elements = newJoinedSelector.elements.concat(parentSel[0].elements.slice(1));
                            }

                            if (!newJoinedSelectorEmpty) {
                                // now add the joined selector
                                newSelectorPath.push(newJoinedSelector);
                            }

                            // and the rest of the parent
                            newSelectorPath = newSelectorPath.concat(afterParentJoin);

                            // add that to our new set of selectors
                            selectorsMultiplied.push(newSelectorPath);
                        }
                    }
                }

                // our new selectors has been multiplied, so reset the state
                newSelectors = selectorsMultiplied;
                currentElements = [];
            }
        }

        // if we have any elements left over (e.g. .a& .b == .b)
        // add them on to all the current selectors
        if (currentElements.length > 0) {
            this.mergeElementsOnToSelectors(currentElements, newSelectors);
        }

        for (i = 0; i < newSelectors.length; i++) {
            if (newSelectors[i].length > 0) {
                paths.push(newSelectors[i]);
            }
        }
    },
    
    mergeElementsOnToSelectors: function(elements, selectors) {
        var i, sel;

        if (selectors.length === 0) {
            selectors.push([ new(tree.Selector)(elements) ]);
            return;
        }

        for (i = 0; i < selectors.length; i++) {
            sel = selectors[i];

            // if the previous thing in sel is a parent this needs to join on to it
            if (sel.length > 0) {
                sel[sel.length - 1] = sel[sel.length - 1].createDerived(sel[sel.length - 1].elements.concat(elements));
            }
            else {
                sel.push(new(tree.Selector)(elements));
            }
        }
    }
};
})(require('../tree'));
