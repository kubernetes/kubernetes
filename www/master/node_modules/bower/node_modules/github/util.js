/** section: github
 * class Util
 * 
 *  Copyright 2012 Cloud9 IDE, Inc.
 *
 *  This product includes software developed by
 *  Cloud9 IDE, Inc (http://c9.io).
 *
 *  Author: Mike de Boer <mike@c9.io>
 **/

var Util = require("util");

/**
 *  Util#extend(dest, src, noOverwrite) -> Object
 *      - dest (Object): destination object
 *      - src (Object): source object
 *      - noOverwrite (Boolean): set to `true` to overwrite values in `src`
 *
 *  Shallow copy of properties from the `src` object to the `dest` object. If the
 *  `noOverwrite` argument is set to to `true`, the value of a property in `src`
 *  will not be overwritten if it already exists.
 **/
exports.extend = function(dest, src, noOverwrite) {
    for (var prop in src) {
        if (!noOverwrite || typeof dest[prop] == "undefined")
            dest[prop] = src[prop];
    }
    return dest;
};

/**
 *  Util#escapeRegExp(str) -> String
 *      - str (String): string to escape
 * 
 *  Escapes characters inside a string that will an error when it is used as part
 *  of a regex upon instantiation like in `new RegExp("[0-9" + str + "]")`
 **/
exports.escapeRegExp = function(str) {
    return str.replace(/([.*+?^${}()|[\]\/\\])/g, '\\$1');
};

/**
 *  Util#toCamelCase(str, [upper]) -> String
 *      - str (String): string to transform
 *      - upper (Boolean): set to `true` to transform to CamelCase
 * 
 *  Transform a string that contains spaces or dashes to camelCase. If `upper` is
 *  set to `true`, the string will be transformed to CamelCase.
 * 
 *  Example:
 *  
 *      Util.toCamelCase("why U no-work"); // returns 'whyUNoWork'
 *      Util.toCamelCase("I U no-work", true); // returns 'WhyUNoWork'
 **/
exports.toCamelCase = function(str, upper) {
    str = str.toLowerCase().replace(/(?:(^.)|(\s+.)|(-.))/g, function(match) {
        return match.charAt(match.length - 1).toUpperCase();
    });
    if (upper)
        return str;
    return str.charAt(0).toLowerCase() + str.substr(1);
};

/**
 *  Util#isTrue(c) -> Boolean
 *      - c (mixed): value the variable to check. Possible values:
 *          true   The function returns true.
 *          'true' The function returns true.
 *          'on'   The function returns true.
 *          1      The function returns true.
 *          '1'    The function returns true.
 * 
 *  Determines whether a string is true in the html attribute sense.
 **/
exports.isTrue = function(c){
    return (c === true || c === "true" || c === "on" || typeof c == "number" && c > 0 || c === "1");
};

/**
 *  Util#isFalse(c) -> Boolean
 *      - c (mixed): value the variable to check. Possible values:
 *          false   The function returns true.
 *          'false' The function returns true.
 *          'off'   The function returns true.
 *          0       The function returns true.
 *          '0'     The function returns true.
 * 
 *  Determines whether a string is false in the html attribute sense.
 **/
exports.isFalse = function(c){
    return (c === false || c === "false" || c === "off" || c === 0 || c === "0");
};

var levels = {
    "info":  ["\033[90m", "\033[39m"], // grey
    "error": ["\033[31m", "\033[39m"], // red
    "fatal": ["\033[35m", "\033[39m"], // magenta
    "exit":  ["\033[36m", "\033[39m"]  // cyan
};
var _slice = Array.prototype.slice;

/**
 *  Util#log(arg1, [arg2], [type]) -> null
 *      - arg1 (mixed): messages to be printed to the standard output
 *      - type (String): type denotation of the message. Possible values:
 *          'info', 'error', 'fatal', 'exit'. Optional, defaults to 'info'.
 * 
 *  Unified logging to the console; arguments passed to this function will put logged
 *  to the standard output of the current process and properly formatted.
 *  Any non-String object will be inspected by the NodeJS util#inspect utility
 *  function.
 *  Messages will be prefixed with its type (with corresponding font color), like so:
 * 
 *      [info] informational message
 *      [error] error message
 *      [fatal] fatal error message
 *      [exit] program exit message (not an error)
 * 
 * The type of message can be defined by passing it to this function as the last/ 
 * final argument. If the type can not be found, this last/ final argument will be
 * regarded as yet another message.
 **/
exports.log = function() {
    var args = _slice.call(arguments);
    var lastArg = args[args.length - 1];

    var level = levels[lastArg] ? args.pop() : "info";
    if (!args.length)
        return;

    var msg = args.map(function(arg) {
        return typeof arg != "string" ? Util.inspect(arg) : arg;
    }).join(" ");
    var pfx = levels[level][0] + "[" + level + "]" + levels[level][1];

    msg.split("\n").forEach(function(line) {
        console.log(pfx + " " + line);
    });
};
