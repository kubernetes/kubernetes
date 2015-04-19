var ctx;
module.exports = {
    init: function(_ctx) {
        console.error("plugin init called, got ctx with keys " + Object.keys(_ctx));

        // ctx contains a bunch of helpers and data
        // stash it away so you can use it inside match
        ctx = _ctx;

        // if you want to setup position triggers now, checkout nginject-comments.js
    },
    match: function(node) {
        console.error("plugin match called, node with type " + node.type);

        // if you think you have a match, return the found target node
        // (may or may not be the passed in argument node)
        // you may also return an array of target nodes

        // ng-annotate will then execute replaceRemoveOrInsertArrayForTarget
        // on every target, i.e. it may remove an array (if --remove) and it may
        // add an array (if --add)

        // please consider filing an issue if you need to workaround a defect or
        // an obviously missing feature in ng-annotate. we'll try to fix it!

        // you know about /* @ngInject */, don't you? (you may not need a plugin)

        // please consider sending a pull request if your plugin is of general use
    },
};
