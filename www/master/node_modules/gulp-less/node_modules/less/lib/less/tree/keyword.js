(function (tree) {

tree.Keyword = function (value) { this.value = value; };
tree.Keyword.prototype = {
    type: "Keyword",
    eval: function () { return this; },
    genCSS: function (env, output) {
        if (this.value === '%') { throw { type: "Syntax", message: "Invalid % without number" }; }
        output.add(this.value);
    },
    toCSS: tree.toCSS,
    compare: function (other) {
        if (other instanceof tree.Keyword) {
            return other.value === this.value ? 0 : 1;
        } else {
            return -1;
        }
    }
};

tree.True = new(tree.Keyword)('true');
tree.False = new(tree.Keyword)('false');

})(require('../tree'));
