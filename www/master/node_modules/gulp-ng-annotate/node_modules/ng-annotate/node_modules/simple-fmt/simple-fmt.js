// simple-fmt.js
// MIT licensed, see LICENSE file
// Copyright (c) 2013 Olov Lassus <olov.lassus@gmail.com>

var fmt = (function() {
    "use strict";

    function fmt(str, var_args) {
        var args = Array.prototype.slice.call(arguments, 1);
        return str.replace(/\{(\d+)\}/g, function(s, match) {
            return (match in args ? args[match] : s);
        });
    }

    function obj(str, obj) {
        return str.replace(/\{([_$a-zA-Z0-9][_$a-zA-Z0-9]*)\}/g, function(s, match) {
            return (match in obj ? obj[match] : s);
        });
    }

    function repeat(str, n) {
        return (new Array(n + 1)).join(str);
    }

    fmt.fmt = fmt;
    fmt.obj = obj;
    fmt.repeat = repeat;
    return fmt;
})();

if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
    module.exports = fmt;
}
