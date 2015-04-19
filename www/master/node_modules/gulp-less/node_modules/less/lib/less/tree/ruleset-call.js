(function (tree) {

tree.RulesetCall = function (variable) {
    this.variable = variable;
};
tree.RulesetCall.prototype = {
    type: "RulesetCall",
    accept: function (visitor) {
    },
    eval: function (env) {
        var detachedRuleset = new(tree.Variable)(this.variable).eval(env);
        return detachedRuleset.callEval(env);
    }
};

})(require('../tree'));
