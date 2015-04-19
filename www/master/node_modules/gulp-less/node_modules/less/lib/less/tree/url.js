(function (tree) {

tree.URL = function (val, currentFileInfo, isEvald) {
    this.value = val;
    this.currentFileInfo = currentFileInfo;
    this.isEvald = isEvald;
};
tree.URL.prototype = {
    type: "Url",
    accept: function (visitor) {
        this.value = visitor.visit(this.value);
    },
    genCSS: function (env, output) {
        output.add("url(");
        this.value.genCSS(env, output);
        output.add(")");
    },
    toCSS: tree.toCSS,
    eval: function (ctx) {
        var val = this.value.eval(ctx),
            rootpath;

        if (!this.isEvald) {
            // Add the base path if the URL is relative
            rootpath = this.currentFileInfo && this.currentFileInfo.rootpath;
            if (rootpath && typeof val.value === "string" && ctx.isPathRelative(val.value)) {
                if (!val.quote) {
                    rootpath = rootpath.replace(/[\(\)'"\s]/g, function(match) { return "\\"+match; });
                }
                val.value = rootpath + val.value;
            }
            
            val.value = ctx.normalizePath(val.value);

            // Add url args if enabled
            if (ctx.urlArgs) {
                if (!val.value.match(/^\s*data:/)) {
                    var delimiter = val.value.indexOf('?') === -1 ? '?' : '&';
                    var urlArgs = delimiter + ctx.urlArgs;
                    if (val.value.indexOf('#') !== -1) {
                        val.value = val.value.replace('#', urlArgs + '#');
                    } else {
                        val.value += urlArgs;
                    }
                }
            }
        }

        return new(tree.URL)(val, this.currentFileInfo, true);
    }
};

})(require('../tree'));
