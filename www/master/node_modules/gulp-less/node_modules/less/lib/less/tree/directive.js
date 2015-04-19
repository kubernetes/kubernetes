(function (tree) {

tree.Directive = function (name, value, rules, index, currentFileInfo, debugInfo) {
    this.name  = name;
    this.value = value;
    if (rules) {
        this.rules = rules;
        this.rules.allowImports = true;
    }
    this.index = index;
    this.currentFileInfo = currentFileInfo;
    this.debugInfo = debugInfo;
};

tree.Directive.prototype = {
    type: "Directive",
    accept: function (visitor) {
        var value = this.value, rules = this.rules;
        if (rules) {
            rules = visitor.visit(rules);
        }
        if (value) {
            value = visitor.visit(value);
        }
    },
    isRulesetLike: function() {
        return !this.isCharset();
    },
    isCharset: function() {
        return "@charset" === this.name;
    },
    genCSS: function (env, output) {
        var value = this.value, rules = this.rules;
        output.add(this.name, this.currentFileInfo, this.index);
        if (value) {
            output.add(' ');
            value.genCSS(env, output);
        }
        if (rules) {
            tree.outputRuleset(env, output, [rules]);
        } else {
            output.add(';');
        }
    },
    toCSS: tree.toCSS,
    eval: function (env) {
        var value = this.value, rules = this.rules;
        if (value) {
            value = value.eval(env);
        }
        if (rules) {
            rules = rules.eval(env);
            rules.root = true;
        }
        return new(tree.Directive)(this.name, value, rules,
            this.index, this.currentFileInfo, this.debugInfo);
    },
    variable: function (name) { if (this.rules) return tree.Ruleset.prototype.variable.call(this.rules, name); },
    find: function () { if (this.rules) return tree.Ruleset.prototype.find.apply(this.rules, arguments); },
    rulesets: function () { if (this.rules) return tree.Ruleset.prototype.rulesets.apply(this.rules); },
    markReferenced: function () {
        var i, rules;
        this.isReferenced = true;
        if (this.rules) {
            rules = this.rules.rules;
            for (i = 0; i < rules.length; i++) {
                if (rules[i].markReferenced) {
                    rules[i].markReferenced();
                }
            }
        }
    }
};

})(require('../tree'));
