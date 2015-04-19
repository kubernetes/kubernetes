(function (tree) {

tree.Rule = function (name, value, important, merge, index, currentFileInfo, inline, variable) {
    this.name = name;
    this.value = (value instanceof tree.Value || value instanceof tree.Ruleset) ? value : new(tree.Value)([value]);
    this.important = important ? ' ' + important.trim() : '';
    this.merge = merge;
    this.index = index;
    this.currentFileInfo = currentFileInfo;
    this.inline = inline || false;
    this.variable = (variable !== undefined) ? variable
        : (name.charAt && (name.charAt(0) === '@'));
};

tree.Rule.prototype = {
    type: "Rule",
    accept: function (visitor) {
        this.value = visitor.visit(this.value);
    },
    genCSS: function (env, output) {
        output.add(this.name + (env.compress ? ':' : ': '), this.currentFileInfo, this.index);
        try {
            this.value.genCSS(env, output);
        }
        catch(e) {
            e.index = this.index;
            e.filename = this.currentFileInfo.filename;
            throw e;
        }
        output.add(this.important + ((this.inline || (env.lastRule && env.compress)) ? "" : ";"), this.currentFileInfo, this.index);
    },
    toCSS: tree.toCSS,
    eval: function (env) {
        var strictMathBypass = false, name = this.name, variable = this.variable, evaldValue;
        if (typeof name !== "string") {
            // expand 'primitive' name directly to get
            // things faster (~10% for benchmark.less):
            name = (name.length === 1) 
                && (name[0] instanceof tree.Keyword)
                    ? name[0].value : evalName(env, name);
            variable = false; // never treat expanded interpolation as new variable name
        }
        if (name === "font" && !env.strictMath) {
            strictMathBypass = true;
            env.strictMath = true;
        }
        try {
            evaldValue = this.value.eval(env);
            
            if (!this.variable && evaldValue.type === "DetachedRuleset") {
                throw { message: "Rulesets cannot be evaluated on a property.",
                        index: this.index, filename: this.currentFileInfo.filename };
            }

            return new(tree.Rule)(name,
                              evaldValue,
                              this.important,
                              this.merge,
                              this.index, this.currentFileInfo, this.inline,
                              variable);
        }
        catch(e) {
            if (typeof e.index !== 'number') {
                e.index = this.index;
                e.filename = this.currentFileInfo.filename;
            }
            throw e;
        }
        finally {
            if (strictMathBypass) {
                env.strictMath = false;
            }
        }
    },
    makeImportant: function () {
        return new(tree.Rule)(this.name,
                              this.value,
                              "!important",
                              this.merge,
                              this.index, this.currentFileInfo, this.inline);
    }
};

function evalName(env, name) {
    var value = "", i, n = name.length,
        output = {add: function (s) {value += s;}};
    for (i = 0; i < n; i++) {
        name[i].eval(env).genCSS(env, output);
    }
    return value;
}

})(require('../tree'));
