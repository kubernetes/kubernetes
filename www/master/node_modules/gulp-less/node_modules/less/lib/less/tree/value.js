(function (tree) {

tree.Value = function (value) {
    this.value = value;
};
tree.Value.prototype = {
    type: "Value",
    accept: function (visitor) {
        if (this.value) {
            this.value = visitor.visitArray(this.value);
        }
    },
    eval: function (env) {
        if (this.value.length === 1) {
            return this.value[0].eval(env);
        } else {
            return new(tree.Value)(this.value.map(function (v) {
                return v.eval(env);
            }));
        }
    },
    genCSS: function (env, output) {
        var i;
        for(i = 0; i < this.value.length; i++) {
            this.value[i].genCSS(env, output);
            if (i+1 < this.value.length) {
                output.add((env && env.compress) ? ',' : ', ');
            }
        }
    },
    toCSS: tree.toCSS
};

})(require('../tree'));
