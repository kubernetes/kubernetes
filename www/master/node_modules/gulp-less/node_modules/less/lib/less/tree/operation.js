(function (tree) {

tree.Operation = function (op, operands, isSpaced) {
    this.op = op.trim();
    this.operands = operands;
    this.isSpaced = isSpaced;
};
tree.Operation.prototype = {
    type: "Operation",
    accept: function (visitor) {
        this.operands = visitor.visit(this.operands);
    },
    eval: function (env) {
        var a = this.operands[0].eval(env),
            b = this.operands[1].eval(env);

        if (env.isMathOn()) {
            if (a instanceof tree.Dimension && b instanceof tree.Color) {
                a = a.toColor();
            }
            if (b instanceof tree.Dimension && a instanceof tree.Color) {
                b = b.toColor();
            }
            if (!a.operate) {
                throw { type: "Operation",
                        message: "Operation on an invalid type" };
            }

            return a.operate(env, this.op, b);
        } else {
            return new(tree.Operation)(this.op, [a, b], this.isSpaced);
        }
    },
    genCSS: function (env, output) {
        this.operands[0].genCSS(env, output);
        if (this.isSpaced) {
            output.add(" ");
        }
        output.add(this.op);
        if (this.isSpaced) {
            output.add(" ");
        }
        this.operands[1].genCSS(env, output);
    },
    toCSS: tree.toCSS
};

tree.operate = function (env, op, a, b) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return a / b;
    }
};

})(require('../tree'));
