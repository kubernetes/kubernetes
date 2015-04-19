// stringset.js
// MIT licensed, see LICENSE file
// Copyright (c) 2013 Olov Lassus <olov.lassus@gmail.com>

var StringSet = (function() {
    "use strict";

    // to save us a few characters
    var hasOwnProperty = Object.prototype.hasOwnProperty;

    var create = (function() {
        function hasOwnEnumerableProps(obj) {
            for (var prop in obj) {
                if (hasOwnProperty.call(obj, prop)) {
                    return true;
                }
            }
            return false;
        }

        // FF <= 3.6:
        // o = {}; o.hasOwnProperty("__proto__" or "__count__" or "__parent__") => true
        // o = {"__proto__": null}; Object.prototype.hasOwnProperty.call(o, "__proto__" or "__count__" or "__parent__") => false
        function hasOwnPollutedProps(obj) {
            return hasOwnProperty.call(obj, "__count__") || hasOwnProperty.call(obj, "__parent__");
        }

        var useObjectCreate = false;
        if (typeof Object.create === "function") {
            if (!hasOwnEnumerableProps(Object.create(null))) {
                useObjectCreate = true;
            }
        }
        if (useObjectCreate === false) {
            if (hasOwnEnumerableProps({})) {
                throw new Error("StringSet environment error 0, please file a bug at https://github.com/olov/stringset/issues");
            }
        }
        // no throw yet means we can create objects without own enumerable props (safe-guard against VMs and shims)

        var o = (useObjectCreate ? Object.create(null) : {});
        var useProtoClear = false;
        if (hasOwnPollutedProps(o)) {
            o.__proto__ = null;
            if (hasOwnEnumerableProps(o) || hasOwnPollutedProps(o)) {
                throw new Error("StringSet environment error 1, please file a bug at https://github.com/olov/stringset/issues");
            }
            useProtoClear = true;
        }
        // no throw yet means we can create objects without own polluted props (safe-guard against VMs and shims)

        return function() {
            var o = (useObjectCreate ? Object.create(null) : {});
            if (useProtoClear) {
                o.__proto__ = null;
            }
            return o;
        };
    })();

    // stringset ctor
    function stringset(optional_array) {
        // use with or without new
        if (!(this instanceof stringset)) {
            return new stringset(optional_array);
        }
        this.obj = create();
        this.hasProto = false; // false (no __proto__ item) or true (has __proto__ item)

        if (optional_array) {
            this.addMany(optional_array);
        }
    };

    // primitive methods that deals with data representation
    stringset.prototype.has = function(item) {
        // The type-check of item in has, get, set and delete is important because otherwise an object
        // {toString: function() { return "__proto__"; }} can avoid the item === "__proto__" test.
        // The alternative to type-checking would be to force string conversion, i.e. item = String(item);
        if (typeof item !== "string") {
            throw new Error("StringSet expected string item");
        }
        return (item === "__proto__" ?
            this.hasProto :
            hasOwnProperty.call(this.obj, item));
    };

    stringset.prototype.add = function(item) {
        if (typeof item !== "string") {
            throw new Error("StringSet expected string item");
        }
        if (item === "__proto__") {
            this.hasProto = true;
        } else {
            this.obj[item] = true;
        }
    };

    stringset.prototype.remove = function(item) {
        if (typeof item !== "string") {
            throw new Error("StringSet expected string item");
        }
        var didExist = this.has(item);
        if (item === "__proto__") {
            this.hasProto = false;
        } else {
            delete this.obj[item];
        }
        return didExist;
    };

    // alias remove to delete but beware:
    // ss.delete("key"); // OK in ES5 and later
    // ss['delete']("key"); // OK in all ES versions
    // ss.remove("key"); // OK in all ES versions
    stringset.prototype['delete'] = stringset.prototype.remove;

    stringset.prototype.isEmpty = function() {
        for (var item in this.obj) {
            if (hasOwnProperty.call(this.obj, item)) {
                return false;
            }
        }
        return !this.hasProto;
    };

    stringset.prototype.size = function() {
        var len = 0;
        for (var item in this.obj) {
            if (hasOwnProperty.call(this.obj, item)) {
                ++len;
            }
        }
        return (this.hasProto ? len + 1 : len);
    };

    stringset.prototype.items = function() {
        var items = [];
        for (var item in this.obj) {
            if (hasOwnProperty.call(this.obj, item)) {
                items.push(item);
            }
        }
        if (this.hasProto) {
            items.push("__proto__");
        }
        return items;
    };


    // methods that rely on the above primitives
    stringset.prototype.addMany = function(items) {
        if (!Array.isArray(items)) {
            throw new Error("StringSet expected array");
        }
        for (var i = 0; i < items.length; i++) {
            this.add(items[i]);
        }
        return this;
    };

    stringset.prototype.merge = function(other) {
        this.addMany(other.items());
        return this;
    };

    stringset.prototype.clone = function() {
        var other = stringset();
        return other.merge(this);
    };

    stringset.prototype.toString = function() {
        return "{" + this.items().map(JSON.stringify).join(",") + "}";
    };

    return stringset;
})();

if (typeof module !== "undefined" && typeof module.exports !== "undefined") {
    module.exports = StringSet;
}
