(function (tree) {

tree.Negative = function (node) {
    this.value = node;
};
tree.Negative.prototype = {
    type: "Negative",
    accept: function (visitor) {
        this.value = visitor.visit(this.value);
    },
    genCSS: function (env, output) {
        output.add('-');
        this.value.genCSS(env, output);
    },
    toCSS: tree.toCSS,
    eval: function (env) {
        if (env.isMathOn()) {
            return (new(tree.Operation)('*', [new(tree.Dimension)(-1), this.value])).eval(env);
        }
        return new(tree.Negative)(this.value.eval(env));
    }
};

})(require('../tree'));
