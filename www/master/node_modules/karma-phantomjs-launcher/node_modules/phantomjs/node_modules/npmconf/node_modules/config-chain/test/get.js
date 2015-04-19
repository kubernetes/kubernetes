var cc = require("../");

var chain = cc()
    , name = "forFun";

chain
    .add({
        __sample:"for fun only"
    }, name)
    .on("load", function() {
        //It throw exception here
        console.log(chain.get("__sample", name));
        //But if I drop the name param, it run normally and return as expected: "for fun only"
        //console.log(chain.get("__sample"));
    });
