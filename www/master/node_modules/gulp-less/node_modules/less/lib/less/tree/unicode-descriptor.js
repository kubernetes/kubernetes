(function (tree) {

tree.UnicodeDescriptor = function (value) {
    this.value = value;
};
tree.UnicodeDescriptor.prototype = {
    type: "UnicodeDescriptor",
    genCSS: function (env, output) {
        output.add(this.value);
    },
    toCSS: tree.toCSS,
    eval: function () { return this; }
};

})(require('../tree'));
