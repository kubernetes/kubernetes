  // Utilities
  if (!Function.prototype.bind) {
    Function.prototype.bind = function (that) {
      var target = this,
        args = slice.call(arguments, 1);
      var bound = function () {
        if (this instanceof bound) {
          function F() { }
          F.prototype = target.prototype;
          var self = new F();
          var result = target.apply(self, args.concat(slice.call(arguments)));
          if (Object(result) === result) {
            return result;
          }
          return self;
        } else {
          return target.apply(that, args.concat(slice.call(arguments)));
        }
      };

      return bound;
    };
  }

  if (!Array.prototype.forEach) {
    Array.prototype.forEach = function (callback, thisArg) {
      var T, k;

      if (this == null) {
        throw new TypeError(" this is null or not defined");
      }

      var O = Object(this);
      var len = O.length >>> 0;

      if (typeof callback !== "function") {
        throw new TypeError(callback + " is not a function");
      }

      if (arguments.length > 1) {
        T = thisArg;
      }

      k = 0;
      while (k < len) {
        var kValue;
        if (k in O) {
          kValue = O[k];
          callback.call(T, kValue, k, O);
        }
        k++;
      }
    };
  }

  var boxedString = Object("a"),
      splitString = boxedString[0] != "a" || !(0 in boxedString);
  if (!Array.prototype.every) {
    Array.prototype.every = function every(fun /*, thisp */) {
      var object = Object(this),
        self = splitString && {}.toString.call(this) == stringClass ?
          this.split("") :
          object,
        length = self.length >>> 0,
        thisp = arguments[1];

      if ({}.toString.call(fun) != funcClass) {
        throw new TypeError(fun + " is not a function");
      }

      for (var i = 0; i < length; i++) {
        if (i in self && !fun.call(thisp, self[i], i, object)) {
          return false;
        }
      }
      return true;
    };
  }

  if (!Array.prototype.map) {
    Array.prototype.map = function map(fun /*, thisp*/) {
      var object = Object(this),
        self = splitString && {}.toString.call(this) == stringClass ?
            this.split("") :
            object,
        length = self.length >>> 0,
        result = Array(length),
        thisp = arguments[1];

      if ({}.toString.call(fun) != funcClass) {
        throw new TypeError(fun + " is not a function");
      }

      for (var i = 0; i < length; i++) {
        if (i in self) {
          result[i] = fun.call(thisp, self[i], i, object);
        }
      }
      return result;
    };
  }

  if (!Array.prototype.filter) {
    Array.prototype.filter = function (predicate) {
      var results = [], item, t = new Object(this);
      for (var i = 0, len = t.length >>> 0; i < len; i++) {
        item = t[i];
        if (i in t && predicate.call(arguments[1], item, i, t)) {
          results.push(item);
        }
      }
      return results;
    };
  }

  if (!Array.isArray) {
    Array.isArray = function (arg) {
      return {}.toString.call(arg) == arrayClass;
    };
  }

  if (!Array.prototype.indexOf) {
    Array.prototype.indexOf = function indexOf(searchElement) {
      var t = Object(this);
      var len = t.length >>> 0;
      if (len === 0) {
        return -1;
      }
      var n = 0;
      if (arguments.length > 1) {
        n = Number(arguments[1]);
        if (n !== n) {
          n = 0;
        } else if (n !== 0 && n != Infinity && n !== -Infinity) {
          n = (n > 0 || -1) * Math.floor(Math.abs(n));
        }
      }
      if (n >= len) {
        return -1;
      }
      var k = n >= 0 ? n : Math.max(len - Math.abs(n), 0);
      for (; k < len; k++) {
        if (k in t && t[k] === searchElement) {
          return k;
        }
      }
      return -1;
    };
  }

  // Fix for Tessel
  if (!Object.prototype.propertyIsEnumerable) {
    Object.prototype.propertyIsEnumerable = function (key) {
      for (var k in this) { if (k === key) { return true; } }
      return false;
    };
  }

  if (!Object.keys) {
    Object.keys = (function() {
      'use strict';
      var hasOwnProperty = Object.prototype.hasOwnProperty,
      hasDontEnumBug = !({ toString: null }).propertyIsEnumerable('toString');

      return function(obj) {
        if (typeof obj !== 'object' && (typeof obj !== 'function' || obj === null)) {
          throw new TypeError('Object.keys called on non-object');
        }

        var result = [], prop, i;

        for (prop in obj) {
          if (hasOwnProperty.call(obj, prop)) {
            result.push(prop);
          }
        }

        if (hasDontEnumBug) {
          for (i = 0; i < dontEnumsLength; i++) {
            if (hasOwnProperty.call(obj, dontEnums[i])) {
              result.push(dontEnums[i]);
            }
          }
        }
        return result;
      };
    }());
  }
