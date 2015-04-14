/*global module, exports, require*/
/*jslint vars:true, evil:true*/
/* JSONPath 0.8.0 - XPath for JSON
 *
 * Copyright (c) 2007 Stefan Goessner (goessner.net)
 * Licensed under the MIT (MIT-LICENSE.txt) licence.
 */

(function (require) {'use strict';

// Keep compatibility with old browsers
if (!Array.isArray) {
    Array.isArray = function (vArg) {
        return Object.prototype.toString.call(vArg) === '[object Array]';
    };
}

// Make sure to know if we are in real node or not (the `require` variable
// could actually be require.js, for example.
var isNode = typeof module !== 'undefined' && !!module.exports;

var vm = isNode ?
    require('vm') : {
        runInNewContext: function (expr, context) {
            return eval(Object.keys(context).reduce(function (s, vr) {
                return 'var ' + vr + '=' + JSON.stringify(context[vr]) + ';' + s;
            }, expr));
        }
    };

var cache = {};

function push (arr, elem) {arr = arr.slice(); arr.push(elem); return arr;}
function unshift (elem, arr) {arr = arr.slice(); arr.unshift(elem); return arr;}

function JSONPath (opts, obj, expr) {
    if (!(this instanceof JSONPath)) { // Make "new" optional
        return new JSONPath(opts, obj, expr);
    }

    opts = opts || {};
    this.resultType = (opts.resultType && opts.resultType.toLowerCase()) || 'value';
    this.flatten = opts.flatten || false;
    this.wrap = opts.hasOwnProperty('wrap') ? opts.wrap : true;
    this.sandbox = opts.sandbox || {};

    if (opts.autostart !== false) {
        return this.evaluate(obj || opts.json, expr || opts.path);
    }
}

// PUBLIC METHODS

JSONPath.prototype.evaluate = function (obj, expr) {
    var self = this;
    this.obj = obj;
    if (expr && obj && (this.resultType === 'value' || this.resultType === 'path')) {
        var exprList = this._normalize(expr);
        if (exprList[0] === '$' && exprList.length > 1) {exprList.shift();}
        var result = this._trace(exprList, obj, ['$']);
        result = result.filter(function (ea) { return ea && !ea.isParentSelector; });
        if (!result.length) {return this.wrap ? [] : false;}
        if (result.length === 1 && !this.wrap && !Array.isArray(result[0].value)) {return result[0][this.resultType] || false;}
        return result.reduce(function (result, ea) {
            var valOrPath = ea[self.resultType];
            if (self.resultType === 'path') {valOrPath = self._asPath(valOrPath);}
            if (self.flatten && Array.isArray(valOrPath)) {
                result = result.concat(valOrPath);
            } else {
                result.push(valOrPath);
            }
            return result;
        }, []);
    }
};

// PRIVATE METHODS

JSONPath.prototype._normalize = function (expr) {
    if (cache[expr]) {return cache[expr];}
    var subx = [];
    var normalized = expr.replace(/[\['](\??\(.*?\))[\]']/g, function ($0, $1) {return '[#' + (subx.push($1) - 1) + ']';})
                    .replace(/'?\.'?|\['?/g, ';')
                    .replace(/(?:;)?(\^+)(?:;)?/g, function (_, ups) {return ';' + ups.split('').join(';') + ';';})
                    .replace(/;;;|;;/g, ';..;')
                    .replace(/;$|'?\]|'$/g, '');
    var exprList = normalized.split(';').map(function (expr) {
        var match = expr.match(/#([0-9]+)/);
        return !match || !match[1] ? expr : subx[match[1]];
    });
    cache[expr] = exprList;
    return cache[expr];
};

JSONPath.prototype._asPath = function (path) {
    var i, n, x = path, p = '$';
    for (i = 1, n = x.length; i < n; i++) {
        p += /^[0-9*]+$/.test(x[i]) ? ('[' + x[i] + ']') : ("['" + x[i] + "']");
    }
    return p;
};

JSONPath.prototype._trace = function (expr, val, path) {
    // No expr to follow? return path and value as the result of this trace branch
    var self = this;
    if (!expr.length) {return [{path: path, value: val}];}

    var loc = expr[0], x = expr.slice(1);
    // The parent sel computation is handled in the frame above using the
    // ancestor object of val
    if (loc === '^') {return path.length ? [{path: path.slice(0, -1), expr: x, isParentSelector: true}] : [];}

    // We need to gather the return value of recursive trace calls in order to
    // do the parent sel computation.
    var ret = [];
    function addRet (elems) {ret = ret.concat(elems);}

    if (val && val.hasOwnProperty(loc)) { // simple case, directly follow property
        addRet(this._trace(x, val[loc], push(path, loc)));
    }
    else if (loc === '*') { // any property
        this._walk(loc, x, val, path, function (m, l, x, v, p) {
            addRet(self._trace(unshift(m, x), v, p));
        });
    }
    else if (loc === '..') { // all child properties
        addRet(this._trace(x, val, path));
        this._walk(loc, x, val, path, function (m, l, x, v, p) {
            if (typeof v[m] === 'object') {
                addRet(self._trace(unshift('..', x), v[m], push(p, m)));
            }
        });
    }
    else if (loc[0] === '(') { // [(expr)]
        addRet(this._trace(unshift(this._eval(loc, val, path[path.length], path), x), val, path));
    }
    else if (loc.indexOf('?(') === 0) { // [?(expr)]
        this._walk(loc, x, val, path, function (m, l, x, v, p) {
            if (self._eval(l.replace(/^\?\((.*?)\)$/, '$1'), v[m], m, path)) {
                addRet(self._trace(unshift(m, x), v, p));
            }
        });
    }
    else if (loc.indexOf(',') > -1) { // [name1,name2,...]
        var parts, i;
        for (parts = loc.split(','), i = 0; i < parts.length; i++) {
            addRet(this._trace(unshift(parts[i], x), val, path));
        }
    }
    else if (/^(-?[0-9]*):(-?[0-9]*):?([0-9]*)$/.test(loc)) { // [start:end:step]  Python slice syntax
        addRet(this._slice(loc, x, val, path));
    }

    // We check the resulting values for parent selections. For parent
    // selections we discard the value object and continue the trace with the
    // current val object
    return ret.reduce(function (all, ea) {
        return all.concat(ea.isParentSelector ? self._trace(ea.expr, val, ea.path) : [ea]);
    }, []);
};

JSONPath.prototype._walk = function (loc, expr, val, path, f) {
    var i, n, m;
    if (Array.isArray(val)) {
        for (i = 0, n = val.length; i < n; i++) {
            f(i, loc, expr, val, path);
        }
    }
    else if (typeof val === 'object') {
        for (m in val) {
            if (val.hasOwnProperty(m)) {
                f(m, loc, expr, val, path);
            }
        }
    }
};

JSONPath.prototype._slice = function (loc, expr, val, path) {
    if (!Array.isArray(val)) {return;}
    var i,
        len = val.length, parts = loc.split(':'),
        start = (parts[0] && parseInt(parts[0], 10)) || 0,
        end = (parts[1] && parseInt(parts[1], 10)) || len,
        step = (parts[2] && parseInt(parts[2], 10)) || 1;
    start = (start < 0) ? Math.max(0, start + len) : Math.min(len, start);
    end    = (end < 0)    ? Math.max(0, end + len) : Math.min(len, end);
    var ret = [];
    for (i = start; i < end; i += step) {
        ret = ret.concat(this._trace(unshift(i, expr), val, path));
    }
    return ret;
};

JSONPath.prototype._eval = function (code, _v, _vname, path) {
    if (!this.obj || !_v) {return false;}
    if (code.indexOf('@path') > -1) {
        this.sandbox._path = this._asPath(path.concat([_vname]));
        code = code.replace(/@path/g, '_path');
    }
    if (code.indexOf('@') > -1) {
        this.sandbox._v = _v;
        code = code.replace(/@/g, '_v');
    }
    try {
        return vm.runInNewContext(code, this.sandbox);
    }
    catch(e) {
        console.log(e);
        throw new Error('jsonPath: ' + e.message + ': ' + code);
    }
};

// For backward compatibility (deprecated)
JSONPath.eval = function (obj, expr, opts) {
    return new JSONPath(opts, obj, expr);
};

if (typeof module === 'undefined') {
    window.jsonPath = { // Deprecated
        eval: JSONPath.eval
    };
    window.JSONPath = JSONPath;
}
else {
    module.exports = JSONPath;
}

}(typeof require === 'undefined' ? null : require));
