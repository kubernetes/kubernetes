function initRhinoTest() {
    process = { title: 'dummy' };

    less.tree.functions.add = function (a, b) {
        return new(less.tree.Dimension)(a.value + b.value);
    };
    less.tree.functions.increment = function (a) {
        return new(less.tree.Dimension)(a.value + 1);
    };
    less.tree.functions._color = function (str) {
        if (str.value === "evil red") { return new(less.tree.Color)("600"); }
    };
}

initRhinoTest();