(function (tree) {

tree.Quoted = function (str, content, escaped, index, currentFileInfo) {
    this.escaped = escaped;
    this.value = content || '';
    this.quote = str.charAt(0);
    this.index = index;
    this.currentFileInfo = currentFileInfo;
};
tree.Quoted.prototype = {
    type: "Quoted",
    genCSS: function (env, output) {
        if (!this.escaped) {
            output.add(this.quote, this.currentFileInfo, this.index);
        }
        output.add(this.value);
        if (!this.escaped) {
            output.add(this.quote);
        }
    },
    toCSS: tree.toCSS,
    eval: function (env) {
        var that = this;
        var value = this.value.replace(/`([^`]+)`/g, function (_, exp) {
            return new(tree.JavaScript)(exp, that.index, true).eval(env).value;
        }).replace(/@\{([\w-]+)\}/g, function (_, name) {
            var v = new(tree.Variable)('@' + name, that.index, that.currentFileInfo).eval(env, true);
            return (v instanceof tree.Quoted) ? v.value : v.toCSS();
        });
        return new(tree.Quoted)(this.quote + value + this.quote, value, this.escaped, this.index, this.currentFileInfo);
    },
    compare: function (x) {
        if (!x.toCSS) {
            return -1;
        }

        var left, right;

        // when comparing quoted strings allow the quote to differ
        if (x.type === "Quoted" && !this.escaped && !x.escaped) {
            left = x.value;
            right = this.value;
        } else {
            left = this.toCSS();
            right = x.toCSS();
        }

        if (left === right) {
            return 0;
        }

        return left < right ? -1 : 1;
    }
};

})(require('../tree'));
