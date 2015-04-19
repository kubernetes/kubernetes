(function (tree) {

tree.DetachedRuleset = function (ruleset, frames) {
    this.ruleset = ruleset;
    this.frames = frames;
};
tree.DetachedRuleset.prototype = {
    type: "DetachedRuleset",
    accept: function (visitor) {
        this.ruleset = visitor.visit(this.ruleset);
    },
    eval: function (env) {
        var frames = this.frames || env.frames.slice(0);
        return new tree.DetachedRuleset(this.ruleset, frames);
    },
    callEval: function (env) {
        return this.ruleset.eval(this.frames ? new(tree.evalEnv)(env, this.frames.concat(env.frames)) : env);
    }
};
})(require('../tree'));
