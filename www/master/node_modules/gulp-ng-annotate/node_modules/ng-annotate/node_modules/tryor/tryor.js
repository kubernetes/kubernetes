// tryor.js
// MIT licensed, see LICENSE file
// Copyright (c) 2013 Olov Lassus <olov.lassus@gmail.com>

function tryor(fn, v) {
    "use strict";

    try {
        return fn();
    } catch (e) {
        return v;
    }
};

if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
    module.exports = tryor;
}
