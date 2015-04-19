(function (tree) {

tree.Variable = function (name, index, currentFileInfo) {
    this.name = name;
    this.index = index;
    this.currentFileInfo = currentFileInfo || {};
};
tree.Variable.prototype = {
    type: "Variable",
    eval: function (env) {
        var variable, name = this.name;

        if (name.indexOf('@@') === 0) {
            name = '@' + new(tree.Variable)(name.slice(1)).eval(env).value;
        }
        
        if (this.evaluating) {
            throw { type: 'Name',
                    message: "Recursive variable definition for " + name,
                    filename: this.currentFileInfo.file,
                    index: this.index };
        }
        
        this.evaluating = true;

        variable = tree.find(env.frames, function (frame) {
            var v = frame.variable(name);
            if (v) {
                return v.value.eval(env);
            }
        });
        if (variable) { 
            this.evaluating = false;
            return variable;
        } else {
            throw { type: 'Name',
                    message: "variable " + name + " is undefined",
                    filename: this.currentFileInfo.filename,
                    index: this.index };
        }
    }
};

})(require('../tree'));
