"use strict";

var _test = require("tap").test;
var is = require("../");

function test(name, fn) {
    return _test(name, function(t) {
        fn(t, t.ok.bind(t), t.notOk.bind(t));
    });
}

test("various", function(t, yes, no) {
    function fn() {
    }

    yes(is.nan(NaN));
    no(is.nan(1));
    no(is.nan("asdf"));

    yes(is.boolean(true));
    no(is.boolean(new Boolean(true)));

    yes(is.number(1));
    no(is.number(new Number(1)));

    yes(is.string("asdf"));
    no(is.string(new String("asdf")));

    yes(is.fn(fn));
    no(is.fn({}));

    yes(is.object({}));
    no(is.object(null));
    no(is.object(fn));

    yes([null, undefined, true, 1, "asdf"].every(is.primitive));
    no([{}, fn, new Number(1), /regexp/].some(is.primitive));

    yes(is.array([]));

    yes(is.finitenumber(1));
    yes(is.finitenumber(1.1));
    no(is.finitenumber(NaN));
    no(is.finitenumber(Infinity));
    no(is.finitenumber(-Infinity));
    no(is.finitenumber("1"));

    yes(is.someof(1, [-1, 1, 3]));
    yes(is.someof("x", [0, 1, "x"]));
    no(is.someof(1, ["1"]));

    no(is.noneof(1, [-1, 1, 3]));
    no(is.noneof("x", [0, 1, "x"]));
    yes(is.noneof(1, ["1"]));

    yes(is.own({a: 1}, "a"));
    no(is.own({a: 1}, "b"));
    no(is.own({a: 1}, "toString"));

    t.end();
});
