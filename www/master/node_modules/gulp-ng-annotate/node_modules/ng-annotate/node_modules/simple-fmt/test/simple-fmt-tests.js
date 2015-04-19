"use strict";

var test = require("tap").test;
var fmt = require("../");

test("fmt", function(t) {
    t.equals(fmt("all your {0} are belong to {1}", "base", "us"),
        "all your base are belong to us");

    var obj = {
        toString: function() {
            return "yoyoma";
        },
    };

    t.equals(fmt("object is called {0} and is {1} ms old", obj, 1),
        "object is called yoyoma and is 1 ms old");

    t.equals(fmt("no arguments => no modifs {0} {1}"),
        "no arguments => no modifs {0} {1}");

    t.end();
});

test("fmt.obj", function(t) {
    var obj2 = {
        name: "yoyoma",
        age: 1,
    };

    t.equals(fmt.obj("object is called {name} and is {age} ms old", obj2),
        "object is called yoyoma and is 1 ms old");

    t.equals(fmt.obj("no matching properties => no modifs {0} {1} {name} {age}", {}),
        "no matching properties => no modifs {0} {1} {name} {age}");

    t.equals(fmt.obj("works for arrays too: [{2}, {1}, {0}]", ["one", "two", "three"]),
        "works for arrays too: [three, two, one]");

    t.end();
});

test("fmt.repeat", function(t) {
    t.equals(fmt.repeat("*", 3), "***");
    t.equals(fmt.repeat("*", 0), "");
    t.equals(fmt.repeat("", 3), "");

    t.end();
});
