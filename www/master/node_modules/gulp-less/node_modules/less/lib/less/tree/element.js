(function (tree) {

tree.Element = function (combinator, value, index, currentFileInfo) {
    this.combinator = combinator instanceof tree.Combinator ?
                      combinator : new(tree.Combinator)(combinator);

    if (typeof(value) === 'string') {
        this.value = value.trim();
    } else if (value) {
        this.value = value;
    } else {
        this.value = "";
    }
    this.index = index;
    this.currentFileInfo = currentFileInfo;
};
tree.Element.prototype = {
    type: "Element",
    accept: function (visitor) {
        var value = this.value;
        this.combinator = visitor.visit(this.combinator);
        if (typeof value === "object") {
            this.value = visitor.visit(value);
        }
    },
    eval: function (env) {
        return new(tree.Element)(this.combinator,
                                 this.value.eval ? this.value.eval(env) : this.value,
                                 this.index,
                                 this.currentFileInfo);
    },
    genCSS: function (env, output) {
        output.add(this.toCSS(env), this.currentFileInfo, this.index);
    },
    toCSS: function (env) {
        var value = (this.value.toCSS ? this.value.toCSS(env) : this.value);
        if (value === '' && this.combinator.value.charAt(0) === '&') {
            return '';
        } else {
            return this.combinator.toCSS(env || {}) + value;
        }
    }
};

tree.Attribute = function (key, op, value) {
    this.key = key;
    this.op = op;
    this.value = value;
};
tree.Attribute.prototype = {
    type: "Attribute",
    eval: function (env) {
        return new(tree.Attribute)(this.key.eval ? this.key.eval(env) : this.key,
            this.op, (this.value && this.value.eval) ? this.value.eval(env) : this.value);
    },
    genCSS: function (env, output) {
        output.add(this.toCSS(env));
    },
    toCSS: function (env) {
        var value = this.key.toCSS ? this.key.toCSS(env) : this.key;

        if (this.op) {
            value += this.op;
            value += (this.value.toCSS ? this.value.toCSS(env) : this.value);
        }

        return '[' + value + ']';
    }
};

tree.Combinator = function (value) {
    if (value === ' ') {
        this.value = ' ';
    } else {
        this.value = value ? value.trim() : "";
    }
};
tree.Combinator.prototype = {
    type: "Combinator",
    _noSpaceCombinators: {
        '': true,
        ' ': true,
        '|': true
    },
    genCSS: function (env, output) {
        var spaceOrEmpty = (env.compress || this._noSpaceCombinators[this.value]) ? '' : ' ';
        output.add(spaceOrEmpty + this.value + spaceOrEmpty);
    },
    toCSS: tree.toCSS
};

})(require('../tree'));
