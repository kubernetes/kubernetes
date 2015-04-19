//
// LESS - Leaner CSS v1.3.1
// http://lesscss.org
// 
// Copyright (c) 2009-2011, Alexis Sellier
// Licensed under the Apache 2.0 License.
//
(function (window, undefined) {
//
// Stub out `require` in the browser
//
function require(arg) {
    return window.less[arg.split('/')[1]];
};


// ecma-5.js
//
// -- kriskowal Kris Kowal Copyright (C) 2009-2010 MIT License
// -- tlrobinson Tom Robinson
// dantman Daniel Friesen

//
// Array
//
if (!Array.isArray) {
    Array.isArray = function(obj) {
        return Object.prototype.toString.call(obj) === "[object Array]" ||
               (obj instanceof Array);
    };
}
if (!Array.prototype.forEach) {
    Array.prototype.forEach =  function(block, thisObject) {
        var len = this.length >>> 0;
        for (var i = 0; i < len; i++) {
            if (i in this) {
                block.call(thisObject, this[i], i, this);
            }
        }
    };
}
if (!Array.prototype.map) {
    Array.prototype.map = function(fun /*, thisp*/) {
        var len = this.length >>> 0;
        var res = new Array(len);
        var thisp = arguments[1];

        for (var i = 0; i < len; i++) {
            if (i in this) {
                res[i] = fun.call(thisp, this[i], i, this);
            }
        }
        return res;
    };
}
if (!Array.prototype.filter) {
    Array.prototype.filter = function (block /*, thisp */) {
        var values = [];
        var thisp = arguments[1];
        for (var i = 0; i < this.length; i++) {
            if (block.call(thisp, this[i])) {
                values.push(this[i]);
            }
        }
        return values;
    };
}
if (!Array.prototype.reduce) {
    Array.prototype.reduce = function(fun /*, initial*/) {
        var len = this.length >>> 0;
        var i = 0;

        // no value to return if no initial value and an empty array
        if (len === 0 && arguments.length === 1) throw new TypeError();

        if (arguments.length >= 2) {
            var rv = arguments[1];
        } else {
            do {
                if (i in this) {
                    rv = this[i++];
                    break;
                }
                // if array contains no values, no initial value to return
                if (++i >= len) throw new TypeError();
            } while (true);
        }
        for (; i < len; i++) {
            if (i in this) {
                rv = fun.call(null, rv, this[i], i, this);
            }
        }
        return rv;
    };
}
if (!Array.prototype.indexOf) {
    Array.prototype.indexOf = function (value /*, fromIndex */ ) {
        var length = this.length;
        var i = arguments[1] || 0;

        if (!length)     return -1;
        if (i >= length) return -1;
        if (i < 0)       i += length;

        for (; i < length; i++) {
            if (!Object.prototype.hasOwnProperty.call(this, i)) { continue }
            if (value === this[i]) return i;
        }
        return -1;
    };
}

//
// Object
//
if (!Object.keys) {
    Object.keys = function (object) {
        var keys = [];
        for (var name in object) {
            if (Object.prototype.hasOwnProperty.call(object, name)) {
                keys.push(name);
            }
        }
        return keys;
    };
}

//
// String
//
if (!String.prototype.trim) {
    String.prototype.trim = function () {
        return String(this).replace(/^\s\s*/, '').replace(/\s\s*$/, '');
    };
}
var less, tree;

if (typeof environment === "object" && ({}).toString.call(environment) === "[object Environment]") {
    // Rhino
    // Details on how to detect Rhino: https://github.com/ringo/ringojs/issues/88
    if (typeof(window) === 'undefined') { less = {} }
    else                                { less = window.less = {} }
    tree = less.tree = {};
    less.mode = 'rhino';
} else if (typeof(window) === 'undefined') {
    // Node.js
    less = exports,
    tree = require('./tree');
    less.mode = 'node';
} else {
    // Browser
    if (typeof(window.less) === 'undefined') { window.less = {} }
    less = window.less,
    tree = window.less.tree = {};
    less.mode = 'browser';
}
//
// less.js - parser
//
//    A relatively straight-forward predictive parser.
//    There is no tokenization/lexing stage, the input is parsed
//    in one sweep.
//
//    To make the parser fast enough to run in the browser, several
//    optimization had to be made:
//
//    - Matching and slicing on a huge input is often cause of slowdowns.
//      The solution is to chunkify the input into smaller strings.
//      The chunks are stored in the `chunks` var,
//      `j` holds the current chunk index, and `current` holds
//      the index of the current chunk in relation to `input`.
//      This gives us an almost 4x speed-up.
//
//    - In many cases, we don't need to match individual tokens;
//      for example, if a value doesn't hold any variables, operations
//      or dynamic references, the parser can effectively 'skip' it,
//      treating it as a literal.
//      An example would be '1px solid #000' - which evaluates to itself,
//      we don't need to know what the individual components are.
//      The drawback, of course is that you don't get the benefits of
//      syntax-checking on the CSS. This gives us a 50% speed-up in the parser,
//      and a smaller speed-up in the code-gen.
//
//
//    Token matching is done with the `$` function, which either takes
//    a terminal string or regexp, or a non-terminal function to call.
//    It also takes care of moving all the indices forwards.
//
//
less.Parser = function Parser(env) {
    var input,       // LeSS input string
        i,           // current index in `input`
        j,           // current chunk
        temp,        // temporarily holds a chunk's state, for backtracking
        memo,        // temporarily holds `i`, when backtracking
        furthest,    // furthest index the parser has gone to
        chunks,      // chunkified input
        current,     // index of current chunk, in `input`
        parser;

    var that = this;

    // Top parser on an import tree must be sure there is one "env"
    // which will then be passed arround by reference.
    var env = env || { };
    if (!env.contents) { env.contents={}; }  // env.contents must be passed arround with top env

    // This function is called after all files
    // have been imported through `@import`.
    var finish = function () {};

    var imports = this.imports = {
        paths: env && env.paths || [],  // Search paths, when importing
        queue: [],                      // Files which haven't been imported yet
        files: {},                      // Holds the imported parse trees
        contents: env.contents,         // Holds the imported file contents
        mime:  env && env.mime,         // MIME type of .less files
        error: null,                    // Error in parsing/evaluating an import
        push: function (path, callback) {
            var that = this;
            this.queue.push(path);

            //
            // Import a file asynchronously
            //
            less.Parser.importer(path, this.paths, function (e, root) {
                that.queue.splice(that.queue.indexOf(path), 1); // Remove the path from the queue

                var imported = path in that.files;

                that.files[path] = root;                        // Store the root

                if (e && !that.error) { that.error = e }

                callback(e, root, imported);

                if (that.queue.length === 0) { finish(e) }       // Call `finish` if we're done importing
            }, env);
        }
    };

    function save()    { temp = chunks[j], memo = i, current = i }
    function restore() { chunks[j] = temp, i = memo, current = i }

    function sync() {
        if (i > current) {
            chunks[j] = chunks[j].slice(i - current);
            current = i;
        }
    }
    function isWhitespace(c) {
        // Could change to \s?
        var code = c.charCodeAt(0);
        return code === 32 || code === 10 || code === 9;
    }
    //
    // Parse from a token, regexp or string, and move forward if match
    //
    function $(tok) {
        var match, args, length, index, k;

        //
        // Non-terminal
        //
        if (tok instanceof Function) {
            return tok.call(parser.parsers);
        //
        // Terminal
        //
        //     Either match a single character in the input,
        //     or match a regexp in the current chunk (chunk[j]).
        //
        } else if (typeof(tok) === 'string') {
            match = input.charAt(i) === tok ? tok : null;
            length = 1;
            sync ();
        } else {
            sync ();

            if (match = tok.exec(chunks[j])) {
                length = match[0].length;
            } else {
                return null;
            }
        }

        // The match is confirmed, add the match length to `i`,
        // and consume any extra white-space characters (' ' || '\n')
        // which come after that. The reason for this is that LeSS's
        // grammar is mostly white-space insensitive.
        //
        if (match) {
            skipWhitespace(length);

            if(typeof(match) === 'string') {
                return match;
            } else {
                return match.length === 1 ? match[0] : match;
            }
        }
    }

    function skipWhitespace(length) {
        var oldi = i, oldj = j,
            endIndex = i + chunks[j].length,
            mem = i += length;

        while (i < endIndex) {
            if (! isWhitespace(input.charAt(i))) { break }
            i++;
        }
        chunks[j] = chunks[j].slice(length + (i - mem));
        current = i;

        if (chunks[j].length === 0 && j < chunks.length - 1) { j++ }

        return oldi !== i || oldj !== j;
    }

    function expect(arg, msg) {
        var result = $(arg);
        if (! result) {
            error(msg || (typeof(arg) === 'string' ? "expected '" + arg + "' got '" + input.charAt(i) + "'"
                                                   : "unexpected token"));
        } else {
            return result;
        }
    }

    function error(msg, type) {
        throw { index: i, type: type || 'Syntax', message: msg };
    }

    // Same as $(), but don't change the state of the parser,
    // just return the match.
    function peek(tok) {
        if (typeof(tok) === 'string') {
            return input.charAt(i) === tok;
        } else {
            if (tok.test(chunks[j])) {
                return true;
            } else {
                return false;
            }
        }
    }

    function getInput(e, env) {
        if (e.filename && env.filename && (e.filename !== env.filename)) {
            return parser.imports.contents[e.filename];
        } else {
            return input;
        }
    }

    function getLocation(index, input) {
        for (var n = index, column = -1;
                 n >= 0 && input.charAt(n) !== '\n';
                 n--) { column++ }

        return { line:   typeof(index) === 'number' ? (input.slice(0, index).match(/\n/g) || "").length : null,
                 column: column };
    }

    function getFileName(e) {
        if(less.mode === 'browser' || less.mode === 'rhino')
            return e.filename;
        else
            return require('path').resolve(e.filename);
    }

    function getDebugInfo(index, inputStream, e) {
        return {
            lineNumber: getLocation(index, inputStream).line + 1,
            fileName: getFileName(e)
        };
    }

    function LessError(e, env) {
        var input = getInput(e, env),
            loc = getLocation(e.index, input),
            line = loc.line,
            col  = loc.column,
            lines = input.split('\n');

        this.type = e.type || 'Syntax';
        this.message = e.message;
        this.filename = e.filename || env.filename;
        this.index = e.index;
        this.line = typeof(line) === 'number' ? line + 1 : null;
        this.callLine = e.call && (getLocation(e.call, input).line + 1);
        this.callExtract = lines[getLocation(e.call, input).line];
        this.stack = e.stack;
        this.column = col;
        this.extract = [
            lines[line - 1],
            lines[line],
            lines[line + 1]
        ];
    }

    this.env = env = env || {};

    // The optimization level dictates the thoroughness of the parser,
    // the lower the number, the less nodes it will create in the tree.
    // This could matter for debugging, or if you want to access
    // the individual nodes in the tree.
    this.optimization = ('optimization' in this.env) ? this.env.optimization : 1;

    this.env.filename = this.env.filename || null;

    //
    // The Parser
    //
    return parser = {

        imports: imports,
        //
        // Parse an input string into an abstract syntax tree,
        // call `callback` when done.
        //
        parse: function (str, callback) {
            var root, start, end, zone, line, lines, buff = [], c, error = null;

            i = j = current = furthest = 0;
            input = str.replace(/\r\n/g, '\n');

            // Remove potential UTF Byte Order Mark
            input = input.replace(/^\uFEFF/, '');

            // Split the input into chunks.
            chunks = (function (chunks) {
                var j = 0,
                    skip = /(?:@\{[\w-]+\}|[^"'`\{\}\/\(\)\\])+/g,
                    comment = /\/\*(?:[^*]|\*+[^\/*])*\*+\/|\/\/.*/g,
                    string = /"((?:[^"\\\r\n]|\\.)*)"|'((?:[^'\\\r\n]|\\.)*)'|`((?:[^`]|\\.)*)`/g,
                    level = 0,
                    match,
                    chunk = chunks[0],
                    inParam;

                for (var i = 0, c, cc; i < input.length; i++) {
                    skip.lastIndex = i;
                    if (match = skip.exec(input)) {
                        if (match.index === i) {
                            i += match[0].length;
                            chunk.push(match[0]);
                        }
                    }
                    c = input.charAt(i);
                    comment.lastIndex = string.lastIndex = i;

                    if (match = string.exec(input)) {
                        if (match.index === i) {
                            i += match[0].length;
                            chunk.push(match[0]);
                            c = input.charAt(i);
                        }
                    }

                    if (!inParam && c === '/') {
                        cc = input.charAt(i + 1);
                        if (cc === '/' || cc === '*') {
                            if (match = comment.exec(input)) {
                                if (match.index === i) {
                                    i += match[0].length;
                                    chunk.push(match[0]);
                                    c = input.charAt(i);
                                }
                            }
                        }
                    }
                    
                    switch (c) {
                        case '{': if (! inParam) { level ++;        chunk.push(c);                           break }
                        case '}': if (! inParam) { level --;        chunk.push(c); chunks[++j] = chunk = []; break }
                        case '(': if (! inParam) { inParam = true;  chunk.push(c);                           break }
                        case ')': if (  inParam) { inParam = false; chunk.push(c);                           break }
                        default:                                    chunk.push(c);
                    }
                }
                if (level > 0) {
                    error = new(LessError)({
                        index: i,
                        type: 'Parse',
                        message: "missing closing `}`",
                        filename: env.filename
                    }, env);
                }

                return chunks.map(function (c) { return c.join('') });;
            })([[]]);

            if (error) {
                return callback(error);
            }

            // Start with the primary rule.
            // The whole syntax tree is held under a Ruleset node,
            // with the `root` property set to true, so no `{}` are
            // output. The callback is called when the input is parsed.
            try {
                root = new(tree.Ruleset)([], $(this.parsers.primary));
                root.root = true;
            } catch (e) {
                return callback(new(LessError)(e, env));
            }

            root.toCSS = (function (evaluate) {
                var line, lines, column;

                return function (options, variables) {
                    var frames = [], importError;

                    options = options || {};
                    //
                    // Allows setting variables with a hash, so:
                    //
                    //   `{ color: new(tree.Color)('#f01') }` will become:
                    //
                    //   new(tree.Rule)('@color',
                    //     new(tree.Value)([
                    //       new(tree.Expression)([
                    //         new(tree.Color)('#f01')
                    //       ])
                    //     ])
                    //   )
                    //
                    if (typeof(variables) === 'object' && !Array.isArray(variables)) {
                        variables = Object.keys(variables).map(function (k) {
                            var value = variables[k];

                            if (! (value instanceof tree.Value)) {
                                if (! (value instanceof tree.Expression)) {
                                    value = new(tree.Expression)([value]);
                                }
                                value = new(tree.Value)([value]);
                            }
                            return new(tree.Rule)('@' + k, value, false, 0);
                        });
                        frames = [new(tree.Ruleset)(null, variables)];
                    }

                    try {
                        var css = evaluate.call(this, { frames: frames })
                                          .toCSS([], { compress: options.compress || false, dumpLineNumbers: env.dumpLineNumbers });
                    } catch (e) {
                        throw new(LessError)(e, env);
                    }

                    if ((importError = parser.imports.error)) { // Check if there was an error during importing
                        if (importError instanceof LessError) throw importError;
                        else                                  throw new(LessError)(importError, env);
                    }

                    if (options.yuicompress && less.mode === 'node') {
                        return require('./cssmin').compressor.cssmin(css);
                    } else if (options.compress) {
                        return css.replace(/(\s)+/g, "$1");
                    } else {
                        return css;
                    }
                };
            })(root.eval);

            // If `i` is smaller than the `input.length - 1`,
            // it means the parser wasn't able to parse the whole
            // string, so we've got a parsing error.
            //
            // We try to extract a \n delimited string,
            // showing the line where the parse error occured.
            // We split it up into two parts (the part which parsed,
            // and the part which didn't), so we can color them differently.
            if (i < input.length - 1) {
                i = furthest;
                lines = input.split('\n');
                line = (input.slice(0, i).match(/\n/g) || "").length + 1;

                for (var n = i, column = -1; n >= 0 && input.charAt(n) !== '\n'; n--) { column++ }

                error = {
                    type: "Parse",
                    message: "Syntax Error on line " + line,
                    index: i,
                    filename: env.filename,
                    line: line,
                    column: column,
                    extract: [
                        lines[line - 2],
                        lines[line - 1],
                        lines[line]
                    ]
                };
            }

            if (this.imports.queue.length > 0) {
                finish = function (e) {
                    if (e) callback(e);
                    else callback(null, root);
                };
            } else {
                callback(error, root);
            }
        },

        //
        // Here in, the parsing rules/functions
        //
        // The basic structure of the syntax tree generated is as follows:
        //
        //   Ruleset ->  Rule -> Value -> Expression -> Entity
        //
        // Here's some LESS code:
        //
        //    .class {
        //      color: #fff;
        //      border: 1px solid #000;
        //      width: @w + 4px;
        //      > .child {...}
        //    }
        //
        // And here's what the parse tree might look like:
        //
        //     Ruleset (Selector '.class', [
        //         Rule ("color",  Value ([Expression [Color #fff]]))
        //         Rule ("border", Value ([Expression [Dimension 1px][Keyword "solid"][Color #000]]))
        //         Rule ("width",  Value ([Expression [Operation "+" [Variable "@w"][Dimension 4px]]]))
        //         Ruleset (Selector [Element '>', '.child'], [...])
        //     ])
        //
        //  In general, most rules will try to parse a token with the `$()` function, and if the return
        //  value is truly, will return a new node, of the relevant type. Sometimes, we need to check
        //  first, before parsing, that's when we use `peek()`.
        //
        parsers: {
            //
            // The `primary` rule is the *entry* and *exit* point of the parser.
            // The rules here can appear at any level of the parse tree.
            //
            // The recursive nature of the grammar is an interplay between the `block`
            // rule, which represents `{ ... }`, the `ruleset` rule, and this `primary` rule,
            // as represented by this simplified grammar:
            //
            //     primary  →  (ruleset | rule)+
            //     ruleset  →  selector+ block
            //     block    →  '{' primary '}'
            //
            // Only at one point is the primary rule not called from the
            // block rule: at the root level.
            //
            primary: function () {
                var node, root = [];

                while ((node = $(this.mixin.definition) || $(this.rule)    ||  $(this.ruleset) ||
                               $(this.mixin.call)       || $(this.comment) ||  $(this.directive))
                               || $(/^[\s\n]+/)) {
                    node && root.push(node);
                }
                return root;
            },

            // We create a Comment node for CSS comments `/* */`,
            // but keep the LeSS comments `//` silent, by just skipping
            // over them.
            comment: function () {
                var comment;

                if (input.charAt(i) !== '/') return;

                if (input.charAt(i + 1) === '/') {
                    return new(tree.Comment)($(/^\/\/.*/), true);
                } else if (comment = $(/^\/\*(?:[^*]|\*+[^\/*])*\*+\/\n?/)) {
                    return new(tree.Comment)(comment);
                }
            },

            //
            // Entities are tokens which can be found inside an Expression
            //
            entities: {
                //
                // A string, which supports escaping " and '
                //
                //     "milky way" 'he\'s the one!'
                //
                quoted: function () {
                    var str, j = i, e;

                    if (input.charAt(j) === '~') { j++, e = true } // Escaped strings
                    if (input.charAt(j) !== '"' && input.charAt(j) !== "'") return;

                    e && $('~');

                    if (str = $(/^"((?:[^"\\\r\n]|\\.)*)"|'((?:[^'\\\r\n]|\\.)*)'/)) {
                        return new(tree.Quoted)(str[0], str[1] || str[2], e);
                    }
                },

                //
                // A catch-all word, such as:
                //
                //     black border-collapse
                //
                keyword: function () {
                    var k;

                    if (k = $(/^[_A-Za-z-][_A-Za-z0-9-]*/)) {
                        if (tree.colors.hasOwnProperty(k)) {
                            // detect named color
                            return new(tree.Color)(tree.colors[k].slice(1));
                        } else {
                            return new(tree.Keyword)(k);
                        }
                    }
                },

                //
                // A function call
                //
                //     rgb(255, 0, 255)
                //
                // We also try to catch IE's `alpha()`, but let the `alpha` parser
                // deal with the details.
                //
                // The arguments are parsed with the `entities.arguments` parser.
                //
                call: function () {
                    var name, nameLC, args, alpha_ret, index = i;

                    if (! (name = /^([\w-]+|%|progid:[\w\.]+)\(/.exec(chunks[j]))) return;

                    name = name[1];
                    nameLC = name.toLowerCase();

                    if (nameLC === 'url') { return null }
                    else                { i += name.length }

                    if (nameLC === 'alpha') {
                        alpha_ret = $(this.alpha);
                        if(typeof alpha_ret !== 'undefined') {
                            return alpha_ret;
                        }
                    }

                    $('('); // Parse the '(' and consume whitespace.

                    args = $(this.entities.arguments);

                    if (! $(')')) return;

                    if (name) { return new(tree.Call)(name, args, index, env.filename) }
                },
                arguments: function () {
                    var args = [], arg;

                    while (arg = $(this.entities.assignment) || $(this.expression)) {
                        args.push(arg);
                        if (! $(',')) { break }
                    }
                    return args;
                },
                literal: function () {
                    return $(this.entities.ratio) ||
                           $(this.entities.dimension) ||
                           $(this.entities.color) ||
                           $(this.entities.quoted);
                },

                // Assignments are argument entities for calls.
                // They are present in ie filter properties as shown below.
                //
                //     filter: progid:DXImageTransform.Microsoft.Alpha( *opacity=50* )
                //

                assignment: function () {
                    var key, value;
                    if ((key = $(/^\w+(?=\s?=)/i)) && $('=') && (value = $(this.entity))) {
                        return new(tree.Assignment)(key, value);
                    }
                },

                //
                // Parse url() tokens
                //
                // We use a specific rule for urls, because they don't really behave like
                // standard function calls. The difference is that the argument doesn't have
                // to be enclosed within a string, so it can't be parsed as an Expression.
                //
                url: function () {
                    var value;

                    if (input.charAt(i) !== 'u' || !$(/^url\(/)) return;
                    value = $(this.entities.quoted)  || $(this.entities.variable) ||
                            $(/^(?:(?:\\[\(\)'"])|[^\(\)'"])+/) || "";

                    expect(')');

                    return new(tree.URL)((value.value != null || value instanceof tree.Variable)
                                        ? value : new(tree.Anonymous)(value), imports.paths);
                },

                //
                // A Variable entity, such as `@fink`, in
                //
                //     width: @fink + 2px
                //
                // We use a different parser for variable definitions,
                // see `parsers.variable`.
                //
                variable: function () {
                    var name, index = i;

                    if (input.charAt(i) === '@' && (name = $(/^@@?[\w-]+/))) {
                        return new(tree.Variable)(name, index, env.filename);
                    }
                },

                // A variable entity useing the protective {} e.g. @{var}
                variableCurly: function () {
                    var name, curly, index = i;

                    if (input.charAt(i) === '@' && (curly = $(/^@\{([\w-]+)\}/))) {
                        return new(tree.Variable)("@" + curly[1], index, env.filename);
                    }
                },

                //
                // A Hexadecimal color
                //
                //     #4F3C2F
                //
                // `rgb` and `hsl` colors are parsed through the `entities.call` parser.
                //
                color: function () {
                    var rgb;

                    if (input.charAt(i) === '#' && (rgb = $(/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})/))) {
                        return new(tree.Color)(rgb[1]);
                    }
                },

                //
                // A Dimension, that is, a number and a unit
                //
                //     0.5em 95%
                //
                dimension: function () {
                    var value, c = input.charCodeAt(i);
                    if ((c > 57 || c < 45) || c === 47) return;

                    if (value = $(/^(-?\d*\.?\d+)(px|%|em|pc|ex|in|deg|s|ms|pt|cm|mm|rad|grad|turn|dpi|dpcm|dppx|rem|vw|vh|vmin|vm|ch)?/)) {
                        return new(tree.Dimension)(value[1], value[2]);
                    }
                },

                //
                // A Ratio
                //
                //    16/9
                //
                ratio: function () {
                  var value, c = input.charCodeAt(i);
                  if (c > 57 || c < 48) return;

                  if (value = $(/^(\d+\/\d+)/)) {
                    return new(tree.Ratio)(value[1]);
                  }
                },

                //
                // JavaScript code to be evaluated
                //
                //     `window.location.href`
                //
                javascript: function () {
                    var str, j = i, e;

                    if (input.charAt(j) === '~') { j++, e = true } // Escaped strings
                    if (input.charAt(j) !== '`') { return }

                    e && $('~');

                    if (str = $(/^`([^`]*)`/)) {
                        return new(tree.JavaScript)(str[1], i, e);
                    }
                }
            },

            //
            // The variable part of a variable definition. Used in the `rule` parser
            //
            //     @fink:
            //
            variable: function () {
                var name;

                if (input.charAt(i) === '@' && (name = $(/^(@[\w-]+)\s*:/))) { return name[1] }
            },

            //
            // A font size/line-height shorthand
            //
            //     small/12px
            //
            // We need to peek first, or we'll match on keywords and dimensions
            //
            shorthand: function () {
                var a, b;

                if (! peek(/^[@\w.%-]+\/[@\w.-]+/)) return;

                save();

                if ((a = $(this.entity)) && $('/') && (b = $(this.entity))) {
                    return new(tree.Shorthand)(a, b);
                }

                restore();
            },

            //
            // Mixins
            //
            mixin: {
                //
                // A Mixin call, with an optional argument list
                //
                //     #mixins > .square(#fff);
                //     .rounded(4px, black);
                //     .button;
                //
                // The `while` loop is there because mixins can be
                // namespaced, but we only support the child and descendant
                // selector for now.
                //
                call: function () {
                    var elements = [], e, c, args = [], arg, index = i, s = input.charAt(i), name, value, important = false;

                    if (s !== '.' && s !== '#') { return }
                    
                    save(); // stop us absorbing part of an invalid selector

                    while (e = $(/^[#.](?:[\w-]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+/)) {
                        elements.push(new(tree.Element)(c, e, i));
                        c = $('>');
                    }
                    if ($('(')) {
                        while (arg = $(this.expression)) {
                            value = arg;
                            name = null;

                            // Variable
                            if (arg.value.length == 1) {
                                var val = arg.value[0];
                                if (val instanceof tree.Variable) {
                                    if ($(':')) {
                                        if (value = $(this.expression)) {
                                            name = val.name;
                                        } else {
                                            throw new(Error)("Expected value");
                                        }
                                    }
                                }
                            }

                            args.push({ name: name, value: value });

                            if (! $(',')) { break }
                        }
                        if (! $(')')) throw new(Error)("Expected )");
                    }

                    if ($(this.important)) {
                        important = true;
                    }

                    if (elements.length > 0 && ($(';') || peek('}'))) {
                        return new(tree.mixin.Call)(elements, args, index, env.filename, important);
                    }
                    
                    restore();
                },

                //
                // A Mixin definition, with a list of parameters
                //
                //     .rounded (@radius: 2px, @color) {
                //        ...
                //     }
                //
                // Until we have a finer grained state-machine, we have to
                // do a look-ahead, to make sure we don't have a mixin call.
                // See the `rule` function for more information.
                //
                // We start by matching `.rounded (`, and then proceed on to
                // the argument list, which has optional default values.
                // We store the parameters in `params`, with a `value` key,
                // if there is a value, such as in the case of `@radius`.
                //
                // Once we've got our params list, and a closing `)`, we parse
                // the `{...}` block.
                //
                definition: function () {
                    var name, params = [], match, ruleset, param, value, cond, variadic = false;
                    if ((input.charAt(i) !== '.' && input.charAt(i) !== '#') ||
                        peek(/^[^{]*(;|})/)) return;

                    save();

                    if (match = $(/^([#.](?:[\w-]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+)\s*\(/)) {
                        name = match[1];

                        do {
                            if (input.charAt(i) === '.' && $(/^\.{3}/)) {
                                variadic = true;
                                break;
                            } else if (param = $(this.entities.variable) || $(this.entities.literal)
                                                                         || $(this.entities.keyword)) {
                                // Variable
                                if (param instanceof tree.Variable) {
                                    if ($(':')) {
                                        value = expect(this.expression, 'expected expression');
                                        params.push({ name: param.name, value: value });
                                    } else if ($(/^\.{3}/)) {
                                        params.push({ name: param.name, variadic: true });
                                        variadic = true;
                                        break;
                                    } else {
                                        params.push({ name: param.name });
                                    }
                                } else {
                                    params.push({ value: param });
                                }
                            } else {
                                break;
                            }
                        } while ($(','))

                        // .mixincall("@{a}"); 
                        // looks a bit like a mixin definition.. so we have to be nice and restore
                        if (!$(')')) {
                            furthest = i;
                            restore();
                        }

                        if ($(/^when/)) { // Guard
                            cond = expect(this.conditions, 'expected condition');
                        }

                        ruleset = $(this.block);

                        if (ruleset) {
                            return new(tree.mixin.Definition)(name, params, ruleset, cond, variadic);
                        } else {
                            restore();
                        }
                    }
                }
            },

            //
            // Entities are the smallest recognized token,
            // and can be found inside a rule's value.
            //
            entity: function () {
                return $(this.entities.literal) || $(this.entities.variable) || $(this.entities.url) ||
                       $(this.entities.call)    || $(this.entities.keyword)  || $(this.entities.javascript) ||
                       $(this.comment);
            },

            //
            // A Rule terminator. Note that we use `peek()` to check for '}',
            // because the `block` rule will be expecting it, but we still need to make sure
            // it's there, if ';' was ommitted.
            //
            end: function () {
                return $(';') || peek('}');
            },

            //
            // IE's alpha function
            //
            //     alpha(opacity=88)
            //
            alpha: function () {
                var value;

                if (! $(/^\(opacity=/i)) return;
                if (value = $(/^\d+/) || $(this.entities.variable)) {
                    expect(')');
                    return new(tree.Alpha)(value);
                }
            },

            //
            // A Selector Element
            //
            //     div
            //     + h1
            //     #socks
            //     input[type="text"]
            //
            // Elements are the building blocks for Selectors,
            // they are made out of a `Combinator` (see combinator rule),
            // and an element name, such as a tag a class, or `*`.
            //
            element: function () {
                var e, t, c, v;

                c = $(this.combinator);

                e = $(/^(?:\d+\.\d+|\d+)%/) || $(/^(?:[.#]?|:*)(?:[\w-]|[^\x00-\x9f]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+/) ||
                    $('*') || $('&') || $(this.attribute) || $(/^\([^)@]+\)/) || $(/^[\.#](?=@)/) || $(this.entities.variableCurly);

                if (! e) {
                    if ($('(') && (v = ($(this.entities.variableCurly) || $(this.entities.variable))) && $(')')) {
                        e = new(tree.Paren)(v);
                    }
                }

                if (e) { return new(tree.Element)(c, e, i) }
            },

            //
            // Combinators combine elements together, in a Selector.
            //
            // Because our parser isn't white-space sensitive, special care
            // has to be taken, when parsing the descendant combinator, ` `,
            // as it's an empty space. We have to check the previous character
            // in the input, to see if it's a ` ` character. More info on how
            // we deal with this in *combinator.js*.
            //
            combinator: function () {
                var match, c = input.charAt(i);

                if (c === '>' || c === '+' || c === '~') {
                    i++;
                    while (input.charAt(i).match(/\s/)) { i++ }
                    return new(tree.Combinator)(c);
                } else if (input.charAt(i - 1).match(/\s/)) {
                    return new(tree.Combinator)(" ");
                } else {
                    return new(tree.Combinator)(null);
                }
            },

            //
            // A CSS Selector
            //
            //     .class > div + h1
            //     li a:hover
            //
            // Selectors are made out of one or more Elements, see above.
            //
            selector: function () {
                var sel, e, elements = [], c, match;

                // depreciated, will be removed soon
                if ($('(')) {
                    sel = $(this.entity);
                    expect(')');
                    return new(tree.Selector)([new(tree.Element)('', sel, i)]);
                }

                while (e = $(this.element)) {
                    c = input.charAt(i);
                    elements.push(e)
                    if (c === '{' || c === '}' || c === ';' || c === ',') { break }
                }

                if (elements.length > 0) { return new(tree.Selector)(elements) }
            },
            tag: function () {
                return $(/^[A-Za-z][A-Za-z-]*[0-9]?/) || $('*');
            },
            attribute: function () {
                var attr = '', key, val, op;

                if (! $('[')) return;

                if (key = $(/^(?:[_A-Za-z0-9-]|\\.)+/) || $(this.entities.quoted)) {
                    if ((op = $(/^[|~*$^]?=/)) &&
                        (val = $(this.entities.quoted) || $(/^[\w-]+/))) {
                        attr = [key, op, val.toCSS ? val.toCSS() : val].join('');
                    } else { attr = key }
                }

                if (! $(']')) return;

                if (attr) { return "[" + attr + "]" }
            },

            //
            // The `block` rule is used by `ruleset` and `mixin.definition`.
            // It's a wrapper around the `primary` rule, with added `{}`.
            //
            block: function () {
                var content;
                if ($('{') && (content = $(this.primary)) && $('}')) {
                    return content;
                }
            },

            //
            // div, .class, body > p {...}
            //
            ruleset: function () {
                var selectors = [], s, rules, match, debugInfo;
                save();

                if (env.dumpLineNumbers)
                    debugInfo = getDebugInfo(i, input, env);

                while (s = $(this.selector)) {
                    selectors.push(s);
                    $(this.comment);
                    if (! $(',')) { break }
                    $(this.comment);
                }

                if (selectors.length > 0 && (rules = $(this.block))) {
                    var ruleset = new(tree.Ruleset)(selectors, rules, env.strictImports);
                    if (env.dumpLineNumbers)
                        ruleset.debugInfo = debugInfo;
                    return ruleset;
                } else {
                    // Backtrack
                    furthest = i;
                    restore();
                }
            },
            rule: function () {
                var name, value, c = input.charAt(i), important, match;
                save();

                if (c === '.' || c === '#' || c === '&') { return }

                if (name = $(this.variable) || $(this.property)) {
                    if ((name.charAt(0) != '@') && (match = /^([^@+\/'"*`(;{}-]*);/.exec(chunks[j]))) {
                        i += match[0].length - 1;
                        value = new(tree.Anonymous)(match[1]);
                    } else if (name === "font") {
                        value = $(this.font);
                    } else {
                        value = $(this.value);
                    }
                    important = $(this.important);

                    if (value && $(this.end)) {
                        return new(tree.Rule)(name, value, important, memo);
                    } else {
                        furthest = i;
                        restore();
                    }
                }
            },

            //
            // An @import directive
            //
            //     @import "lib";
            //
            // Depending on our environemnt, importing is done differently:
            // In the browser, it's an XHR request, in Node, it would be a
            // file-system operation. The function used for importing is
            // stored in `import`, which we pass to the Import constructor.
            //
            "import": function () {
                var path, features, index = i;
                
                save();
                
                var dir = $(/^@import(?:-(once))?\s+/);

                if (dir && (path = $(this.entities.quoted) || $(this.entities.url))) {
                    features = $(this.mediaFeatures);
                    if ($(';')) {
                        return new(tree.Import)(path, imports, features, (dir[1] === 'once'), index);
                    }
                }
                
                restore();
            },

            mediaFeature: function () {
                var e, p, nodes = [];

                do {
                    if (e = $(this.entities.keyword)) {
                        nodes.push(e);
                    } else if ($('(')) {
                        p = $(this.property);
                        e = $(this.entity);
                        if ($(')')) {
                            if (p && e) {
                                nodes.push(new(tree.Paren)(new(tree.Rule)(p, e, null, i, true)));
                            } else if (e) {
                                nodes.push(new(tree.Paren)(e));
                            } else {
                                return null;
                            }
                        } else { return null }
                    }
                } while (e);

                if (nodes.length > 0) {
                    return new(tree.Expression)(nodes);
                }
            },

            mediaFeatures: function () {
                var e, features = [];

                do {
                  if (e = $(this.mediaFeature)) {
                      features.push(e);
                      if (! $(',')) { break }
                  } else if (e = $(this.entities.variable)) {
                      features.push(e);
                      if (! $(',')) { break }
                  }
                } while (e);

                return features.length > 0 ? features : null;
            },

            media: function () {
                var features, rules, media, debugInfo;

                if (env.dumpLineNumbers)
                    debugInfo = getDebugInfo(i, input, env);

                if ($(/^@media/)) {
                    features = $(this.mediaFeatures);

                    if (rules = $(this.block)) {
                        media = new(tree.Media)(rules, features);
                        if(env.dumpLineNumbers)
                            media.debugInfo = debugInfo;
                        return media;
                    }
                }
            },

            //
            // A CSS Directive
            //
            //     @charset "utf-8";
            //
            directive: function () {
                var name, value, rules, identifier, e, nodes, nonVendorSpecificName,
                    hasBlock, hasIdentifier;

                if (input.charAt(i) !== '@') return;

                if (value = $(this['import']) || $(this.media)) {
                    return value;
                }
                
                save();

                name = $(/^@[a-z-]+/);

                nonVendorSpecificName = name;
                if (name.charAt(1) == '-' && name.indexOf('-', 2) > 0) {
                    nonVendorSpecificName = "@" + name.slice(name.indexOf('-', 2) + 1);
                }

                switch(nonVendorSpecificName) {
                    case "@font-face":
                        hasBlock = true;
                        break;
                    case "@viewport":
                    case "@top-left":
                    case "@top-left-corner":
                    case "@top-center":
                    case "@top-right":
                    case "@top-right-corner":
                    case "@bottom-left":
                    case "@bottom-left-corner":
                    case "@bottom-center":
                    case "@bottom-right":
                    case "@bottom-right-corner":
                    case "@left-top":
                    case "@left-middle":
                    case "@left-bottom":
                    case "@right-top":
                    case "@right-middle":
                    case "@right-bottom":
                        hasBlock = true;
                        break;
                    case "@page":
                    case "@document":
                    case "@supports":
                    case "@keyframes":
                        hasBlock = true;
                        hasIdentifier = true;
                        break;
                }

                if (hasIdentifier) {
                    name += " " + ($(/^[^{]+/) || '').trim();
                }

                if (hasBlock)
                {
                    if (rules = $(this.block)) {
                        return new(tree.Directive)(name, rules);
                    }
                } else {
                    if ((value = $(this.entity)) && $(';')) {
                        return new(tree.Directive)(name, value);
                    }
                }
                
                restore();
            },
            font: function () {
                var value = [], expression = [], weight, shorthand, font, e;

                while (e = $(this.shorthand) || $(this.entity)) {
                    expression.push(e);
                }
                value.push(new(tree.Expression)(expression));

                if ($(',')) {
                    while (e = $(this.expression)) {
                        value.push(e);
                        if (! $(',')) { break }
                    }
                }
                return new(tree.Value)(value);
            },

            //
            // A Value is a comma-delimited list of Expressions
            //
            //     font-family: Baskerville, Georgia, serif;
            //
            // In a Rule, a Value represents everything after the `:`,
            // and before the `;`.
            //
            value: function () {
                var e, expressions = [], important;

                while (e = $(this.expression)) {
                    expressions.push(e);
                    if (! $(',')) { break }
                }

                if (expressions.length > 0) {
                    return new(tree.Value)(expressions);
                }
            },
            important: function () {
                if (input.charAt(i) === '!') {
                    return $(/^! *important/);
                }
            },
            sub: function () {
                var e;

                if ($('(') && (e = $(this.expression)) && $(')')) {
                    return e;
                }
            },
            multiplication: function () {
                var m, a, op, operation;
                if (m = $(this.operand)) {
                    while (!peek(/^\/\*/) && (op = ($('/') || $('*'))) && (a = $(this.operand))) {
                        operation = new(tree.Operation)(op, [operation || m, a]);
                    }
                    return operation || m;
                }
            },
            addition: function () {
                var m, a, op, operation;
                if (m = $(this.multiplication)) {
                    while ((op = $(/^[-+]\s+/) || (!isWhitespace(input.charAt(i - 1)) && ($('+') || $('-')))) &&
                           (a = $(this.multiplication))) {
                        operation = new(tree.Operation)(op, [operation || m, a]);
                    }
                    return operation || m;
                }
            },
            conditions: function () {
                var a, b, index = i, condition;

                if (a = $(this.condition)) {
                    while ($(',') && (b = $(this.condition))) {
                        condition = new(tree.Condition)('or', condition || a, b, index);
                    }
                    return condition || a;
                }
            },
            condition: function () {
                var a, b, c, op, index = i, negate = false;

                if ($(/^not/)) { negate = true }
                expect('(');
                if (a = $(this.addition) || $(this.entities.keyword) || $(this.entities.quoted)) {
                    if (op = $(/^(?:>=|=<|[<=>])/)) {
                        if (b = $(this.addition) || $(this.entities.keyword) || $(this.entities.quoted)) {
                            c = new(tree.Condition)(op, a, b, index, negate);
                        } else {
                            error('expected expression');
                        }
                    } else {
                        c = new(tree.Condition)('=', a, new(tree.Keyword)('true'), index, negate);
                    }
                    expect(')');
                    return $(/^and/) ? new(tree.Condition)('and', c, $(this.condition)) : c;
                }
            },

            //
            // An operand is anything that can be part of an operation,
            // such as a Color, or a Variable
            //
            operand: function () {
                var negate, p = input.charAt(i + 1);

                if (input.charAt(i) === '-' && (p === '@' || p === '(')) { negate = $('-') }
                var o = $(this.sub) || $(this.entities.dimension) ||
                        $(this.entities.color) || $(this.entities.variable) ||
                        $(this.entities.call);
                return negate ? new(tree.Operation)('*', [new(tree.Dimension)(-1), o])
                              : o;
            },

            //
            // Expressions either represent mathematical operations,
            // or white-space delimited Entities.
            //
            //     1px solid black
            //     @var * 2
            //
            expression: function () {
                var e, delim, entities = [], d;

                while (e = $(this.addition) || $(this.entity)) {
                    entities.push(e);
                }
                if (entities.length > 0) {
                    return new(tree.Expression)(entities);
                }
            },
            property: function () {
                var name;

                if (name = $(/^(\*?-?[_a-z0-9-]+)\s*:/)) {
                    return name[1];
                }
            }
        }
    };
};

if (less.mode === 'browser' || less.mode === 'rhino') {
    //
    // Used by `@import` directives
    //
    less.Parser.importer = function (path, paths, callback, env) {
        if (!/^([a-z-]+:)?\//.test(path) && paths.length > 0) {
            path = paths[0] + path;
        }
        // We pass `true` as 3rd argument, to force the reload of the import.
        // This is so we can get the syntax tree as opposed to just the CSS output,
        // as we need this to evaluate the current stylesheet.
        // __ Now using the hack of passing a ref to top parser's content cache in the 1st arg. __
        loadStyleSheet({ href: path, title: path, type: env.mime, contents: env.contents }, function (e) {
            if (e && typeof(env.errback) === "function") {
                env.errback.call(null, path, paths, callback, env);
            } else {
                callback.apply(null, arguments);
            }
        }, true);
    };
}

(function (tree) {

tree.functions = {
    rgb: function (r, g, b) {
        return this.rgba(r, g, b, 1.0);
    },
    rgba: function (r, g, b, a) {
        var rgb = [r, g, b].map(function (c) { return number(c) }),
            a = number(a);
        return new(tree.Color)(rgb, a);
    },
    hsl: function (h, s, l) {
        return this.hsla(h, s, l, 1.0);
    },
    hsla: function (h, s, l, a) {
        h = (number(h) % 360) / 360;
        s = number(s); l = number(l); a = number(a);

        var m2 = l <= 0.5 ? l * (s + 1) : l + s - l * s;
        var m1 = l * 2 - m2;

        return this.rgba(hue(h + 1/3) * 255,
                         hue(h)       * 255,
                         hue(h - 1/3) * 255,
                         a);

        function hue(h) {
            h = h < 0 ? h + 1 : (h > 1 ? h - 1 : h);
            if      (h * 6 < 1) return m1 + (m2 - m1) * h * 6;
            else if (h * 2 < 1) return m2;
            else if (h * 3 < 2) return m1 + (m2 - m1) * (2/3 - h) * 6;
            else                return m1;
        }
    },
    hue: function (color) {
        return new(tree.Dimension)(Math.round(color.toHSL().h));
    },
    saturation: function (color) {
        return new(tree.Dimension)(Math.round(color.toHSL().s * 100), '%');
    },
    lightness: function (color) {
        return new(tree.Dimension)(Math.round(color.toHSL().l * 100), '%');
    },
    red: function (color) {
        return new(tree.Dimension)(color.rgb[0]);
    },
    green: function (color) {
        return new(tree.Dimension)(color.rgb[1]);
    },
    blue: function (color) {
        return new(tree.Dimension)(color.rgb[2]);
    },
    alpha: function (color) {
        return new(tree.Dimension)(color.toHSL().a);
    },
    luma: function (color) {
        return new(tree.Dimension)(Math.round((0.2126 * (color.rgb[0]/255) +
            0.7152 * (color.rgb[1]/255) +
            0.0722 * (color.rgb[2]/255))
            * color.alpha * 100), '%');
    },
    saturate: function (color, amount) {
        var hsl = color.toHSL();

        hsl.s += amount.value / 100;
        hsl.s = clamp(hsl.s);
        return hsla(hsl);
    },
    desaturate: function (color, amount) {
        var hsl = color.toHSL();

        hsl.s -= amount.value / 100;
        hsl.s = clamp(hsl.s);
        return hsla(hsl);
    },
    lighten: function (color, amount) {
        var hsl = color.toHSL();

        hsl.l += amount.value / 100;
        hsl.l = clamp(hsl.l);
        return hsla(hsl);
    },
    darken: function (color, amount) {
        var hsl = color.toHSL();

        hsl.l -= amount.value / 100;
        hsl.l = clamp(hsl.l);
        return hsla(hsl);
    },
    fadein: function (color, amount) {
        var hsl = color.toHSL();

        hsl.a += amount.value / 100;
        hsl.a = clamp(hsl.a);
        return hsla(hsl);
    },
    fadeout: function (color, amount) {
        var hsl = color.toHSL();

        hsl.a -= amount.value / 100;
        hsl.a = clamp(hsl.a);
        return hsla(hsl);
    },
    fade: function (color, amount) {
        var hsl = color.toHSL();

        hsl.a = amount.value / 100;
        hsl.a = clamp(hsl.a);
        return hsla(hsl);
    },
    spin: function (color, amount) {
        var hsl = color.toHSL();
        var hue = (hsl.h + amount.value) % 360;

        hsl.h = hue < 0 ? 360 + hue : hue;

        return hsla(hsl);
    },
    //
    // Copyright (c) 2006-2009 Hampton Catlin, Nathan Weizenbaum, and Chris Eppstein
    // http://sass-lang.com
    //
    mix: function (color1, color2, weight) {
        if (!weight) {
            weight = new(tree.Dimension)(50);
        }
        var p = weight.value / 100.0;
        var w = p * 2 - 1;
        var a = color1.toHSL().a - color2.toHSL().a;

        var w1 = (((w * a == -1) ? w : (w + a) / (1 + w * a)) + 1) / 2.0;
        var w2 = 1 - w1;

        var rgb = [color1.rgb[0] * w1 + color2.rgb[0] * w2,
                   color1.rgb[1] * w1 + color2.rgb[1] * w2,
                   color1.rgb[2] * w1 + color2.rgb[2] * w2];

        var alpha = color1.alpha * p + color2.alpha * (1 - p);

        return new(tree.Color)(rgb, alpha);
    },
    greyscale: function (color) {
        return this.desaturate(color, new(tree.Dimension)(100));
    },
    contrast: function (color, dark, light, threshold) {
        if (typeof light === 'undefined') {
            light = this.rgba(255, 255, 255, 1.0);
        }
        if (typeof dark === 'undefined') {
            dark = this.rgba(0, 0, 0, 1.0);
        }
        if (typeof threshold === 'undefined') {
            threshold = 0.43;
        } else {
            threshold = threshold.value;
        }
        if (((0.2126 * (color.rgb[0]/255) + 0.7152 * (color.rgb[1]/255) + 0.0722 * (color.rgb[2]/255)) * color.alpha) < threshold) {
            return light;
        } else {
            return dark;
        }
    },
    e: function (str) {
        return new(tree.Anonymous)(str instanceof tree.JavaScript ? str.evaluated : str);
    },
    escape: function (str) {
        return new(tree.Anonymous)(encodeURI(str.value).replace(/=/g, "%3D").replace(/:/g, "%3A").replace(/#/g, "%23").replace(/;/g, "%3B").replace(/\(/g, "%28").replace(/\)/g, "%29"));
    },
    '%': function (quoted /* arg, arg, ...*/) {
        var args = Array.prototype.slice.call(arguments, 1),
            str = quoted.value;

        for (var i = 0; i < args.length; i++) {
            str = str.replace(/%[sda]/i, function(token) {
                var value = token.match(/s/i) ? args[i].value : args[i].toCSS();
                return token.match(/[A-Z]$/) ? encodeURIComponent(value) : value;
            });
        }
        str = str.replace(/%%/g, '%');
        return new(tree.Quoted)('"' + str + '"', str);
    },
    round: function (n, f) {
        var fraction = typeof(f) === "undefined" ? 0 : f.value;
        if (n instanceof tree.Dimension) {
            return new(tree.Dimension)(number(n).toFixed(fraction), n.unit);
        } else if (typeof(n) === 'number') {
            return n.toFixed(fraction);
        } else {
            throw { type: "Argument", message: "argument must be a number" };
        }
    },
    ceil: function (n) {
        return this._math('ceil', n);
    },
    floor: function (n) {
        return this._math('floor', n);
    },
    _math: function (fn, n) {
        if (n instanceof tree.Dimension) {
            return new(tree.Dimension)(Math[fn](number(n)), n.unit);
        } else if (typeof(n) === 'number') {
            return Math[fn](n);
        } else {
            throw { type: "Argument", message: "argument must be a number" };
        }
    },
    argb: function (color) {
        return new(tree.Anonymous)(color.toARGB());

    },
    percentage: function (n) {
        return new(tree.Dimension)(n.value * 100, '%');
    },
    color: function (n) {
        if (n instanceof tree.Quoted) {
            return new(tree.Color)(n.value.slice(1));
        } else {
            throw { type: "Argument", message: "argument must be a string" };
        }
    },
    iscolor: function (n) {
        return this._isa(n, tree.Color);
    },
    isnumber: function (n) {
        return this._isa(n, tree.Dimension);
    },
    isstring: function (n) {
        return this._isa(n, tree.Quoted);
    },
    iskeyword: function (n) {
        return this._isa(n, tree.Keyword);
    },
    isurl: function (n) {
        return this._isa(n, tree.URL);
    },
    ispixel: function (n) {
        return (n instanceof tree.Dimension) && n.unit === 'px' ? tree.True : tree.False;
    },
    ispercentage: function (n) {
        return (n instanceof tree.Dimension) && n.unit === '%' ? tree.True : tree.False;
    },
    isem: function (n) {
        return (n instanceof tree.Dimension) && n.unit === 'em' ? tree.True : tree.False;
    },
    _isa: function (n, Type) {
        return (n instanceof Type) ? tree.True : tree.False;
    },
    
    /* Blending modes */
    
    multiply: function(color1, color2) {
        var r = color1.rgb[0] * color2.rgb[0] / 255;
        var g = color1.rgb[1] * color2.rgb[1] / 255;
        var b = color1.rgb[2] * color2.rgb[2] / 255;
        return this.rgb(r, g, b);
    },
    screen: function(color1, color2) {
        var r = 255 - (255 - color1.rgb[0]) * (255 - color2.rgb[0]) / 255;
        var g = 255 - (255 - color1.rgb[1]) * (255 - color2.rgb[1]) / 255;
        var b = 255 - (255 - color1.rgb[2]) * (255 - color2.rgb[2]) / 255;
        return this.rgb(r, g, b);
    },
    overlay: function(color1, color2) {
        var r = color1.rgb[0] < 128 ? 2 * color1.rgb[0] * color2.rgb[0] / 255 : 255 - 2 * (255 - color1.rgb[0]) * (255 - color2.rgb[0]) / 255;
        var g = color1.rgb[1] < 128 ? 2 * color1.rgb[1] * color2.rgb[1] / 255 : 255 - 2 * (255 - color1.rgb[1]) * (255 - color2.rgb[1]) / 255;
        var b = color1.rgb[2] < 128 ? 2 * color1.rgb[2] * color2.rgb[2] / 255 : 255 - 2 * (255 - color1.rgb[2]) * (255 - color2.rgb[2]) / 255;
        return this.rgb(r, g, b);
    },
    softlight: function(color1, color2) {
        var t = color2.rgb[0] * color1.rgb[0] / 255;
        var r = t + color1.rgb[0] * (255 - (255 - color1.rgb[0]) * (255 - color2.rgb[0]) / 255 - t) / 255;
        t = color2.rgb[1] * color1.rgb[1] / 255;
        var g = t + color1.rgb[1] * (255 - (255 - color1.rgb[1]) * (255 - color2.rgb[1]) / 255 - t) / 255;
        t = color2.rgb[2] * color1.rgb[2] / 255;
        var b = t + color1.rgb[2] * (255 - (255 - color1.rgb[2]) * (255 - color2.rgb[2]) / 255 - t) / 255;
        return this.rgb(r, g, b);
    },
    hardlight: function(color1, color2) {
        var r = color2.rgb[0] < 128 ? 2 * color2.rgb[0] * color1.rgb[0] / 255 : 255 - 2 * (255 - color2.rgb[0]) * (255 - color1.rgb[0]) / 255;
        var g = color2.rgb[1] < 128 ? 2 * color2.rgb[1] * color1.rgb[1] / 255 : 255 - 2 * (255 - color2.rgb[1]) * (255 - color1.rgb[1]) / 255;
        var b = color2.rgb[2] < 128 ? 2 * color2.rgb[2] * color1.rgb[2] / 255 : 255 - 2 * (255 - color2.rgb[2]) * (255 - color1.rgb[2]) / 255;
        return this.rgb(r, g, b);
    },
    difference: function(color1, color2) {
        var r = Math.abs(color1.rgb[0] - color2.rgb[0]);
        var g = Math.abs(color1.rgb[1] - color2.rgb[1]);
        var b = Math.abs(color1.rgb[2] - color2.rgb[2]);
        return this.rgb(r, g, b);
    },
    exclusion: function(color1, color2) {
        var r = color1.rgb[0] + color2.rgb[0] * (255 - color1.rgb[0] - color1.rgb[0]) / 255;
        var g = color1.rgb[1] + color2.rgb[1] * (255 - color1.rgb[1] - color1.rgb[1]) / 255;
        var b = color1.rgb[2] + color2.rgb[2] * (255 - color1.rgb[2] - color1.rgb[2]) / 255;
        return this.rgb(r, g, b);
    },
    average: function(color1, color2) {
        var r = (color1.rgb[0] + color2.rgb[0]) / 2;
        var g = (color1.rgb[1] + color2.rgb[1]) / 2;
        var b = (color1.rgb[2] + color2.rgb[2]) / 2;
        return this.rgb(r, g, b);
    },
    negation: function(color1, color2) {
        var r = 255 - Math.abs(255 - color2.rgb[0] - color1.rgb[0]);
        var g = 255 - Math.abs(255 - color2.rgb[1] - color1.rgb[1]);
        var b = 255 - Math.abs(255 - color2.rgb[2] - color1.rgb[2]);
        return this.rgb(r, g, b);
    },
    tint: function(color, amount) {
        return this.mix(this.rgb(255,255,255), color, amount);
    },
    shade: function(color, amount) {
        return this.mix(this.rgb(0, 0, 0), color, amount);
    }
};

function hsla(hsla) {
    return tree.functions.hsla(hsla.h, hsla.s, hsla.l, hsla.a);
}

function number(n) {
    if (n instanceof tree.Dimension) {
        return parseFloat(n.unit == '%' ? n.value / 100 : n.value);
    } else if (typeof(n) === 'number') {
        return n;
    } else {
        throw {
            error: "RuntimeError",
            message: "color functions take numbers as parameters"
        };
    }
}

function clamp(val) {
    return Math.min(1, Math.max(0, val));
}

})(require('./tree'));
(function (tree) {
    tree.colors = {
        'aliceblue':'#f0f8ff',
        'antiquewhite':'#faebd7',
        'aqua':'#00ffff',
        'aquamarine':'#7fffd4',
        'azure':'#f0ffff',
        'beige':'#f5f5dc',
        'bisque':'#ffe4c4',
        'black':'#000000',
        'blanchedalmond':'#ffebcd',
        'blue':'#0000ff',
        'blueviolet':'#8a2be2',
        'brown':'#a52a2a',
        'burlywood':'#deb887',
        'cadetblue':'#5f9ea0',
        'chartreuse':'#7fff00',
        'chocolate':'#d2691e',
        'coral':'#ff7f50',
        'cornflowerblue':'#6495ed',
        'cornsilk':'#fff8dc',
        'crimson':'#dc143c',
        'cyan':'#00ffff',
        'darkblue':'#00008b',
        'darkcyan':'#008b8b',
        'darkgoldenrod':'#b8860b',
        'darkgray':'#a9a9a9',
        'darkgrey':'#a9a9a9',
        'darkgreen':'#006400',
        'darkkhaki':'#bdb76b',
        'darkmagenta':'#8b008b',
        'darkolivegreen':'#556b2f',
        'darkorange':'#ff8c00',
        'darkorchid':'#9932cc',
        'darkred':'#8b0000',
        'darksalmon':'#e9967a',
        'darkseagreen':'#8fbc8f',
        'darkslateblue':'#483d8b',
        'darkslategray':'#2f4f4f',
        'darkslategrey':'#2f4f4f',
        'darkturquoise':'#00ced1',
        'darkviolet':'#9400d3',
        'deeppink':'#ff1493',
        'deepskyblue':'#00bfff',
        'dimgray':'#696969',
        'dimgrey':'#696969',
        'dodgerblue':'#1e90ff',
        'firebrick':'#b22222',
        'floralwhite':'#fffaf0',
        'forestgreen':'#228b22',
        'fuchsia':'#ff00ff',
        'gainsboro':'#dcdcdc',
        'ghostwhite':'#f8f8ff',
        'gold':'#ffd700',
        'goldenrod':'#daa520',
        'gray':'#808080',
        'grey':'#808080',
        'green':'#008000',
        'greenyellow':'#adff2f',
        'honeydew':'#f0fff0',
        'hotpink':'#ff69b4',
        'indianred':'#cd5c5c',
        'indigo':'#4b0082',
        'ivory':'#fffff0',
        'khaki':'#f0e68c',
        'lavender':'#e6e6fa',
        'lavenderblush':'#fff0f5',
        'lawngreen':'#7cfc00',
        'lemonchiffon':'#fffacd',
        'lightblue':'#add8e6',
        'lightcoral':'#f08080',
        'lightcyan':'#e0ffff',
        'lightgoldenrodyellow':'#fafad2',
        'lightgray':'#d3d3d3',
        'lightgrey':'#d3d3d3',
        'lightgreen':'#90ee90',
        'lightpink':'#ffb6c1',
        'lightsalmon':'#ffa07a',
        'lightseagreen':'#20b2aa',
        'lightskyblue':'#87cefa',
        'lightslategray':'#778899',
        'lightslategrey':'#778899',
        'lightsteelblue':'#b0c4de',
        'lightyellow':'#ffffe0',
        'lime':'#00ff00',
        'limegreen':'#32cd32',
        'linen':'#faf0e6',
        'magenta':'#ff00ff',
        'maroon':'#800000',
        'mediumaquamarine':'#66cdaa',
        'mediumblue':'#0000cd',
        'mediumorchid':'#ba55d3',
        'mediumpurple':'#9370d8',
        'mediumseagreen':'#3cb371',
        'mediumslateblue':'#7b68ee',
        'mediumspringgreen':'#00fa9a',
        'mediumturquoise':'#48d1cc',
        'mediumvioletred':'#c71585',
        'midnightblue':'#191970',
        'mintcream':'#f5fffa',
        'mistyrose':'#ffe4e1',
        'moccasin':'#ffe4b5',
        'navajowhite':'#ffdead',
        'navy':'#000080',
        'oldlace':'#fdf5e6',
        'olive':'#808000',
        'olivedrab':'#6b8e23',
        'orange':'#ffa500',
        'orangered':'#ff4500',
        'orchid':'#da70d6',
        'palegoldenrod':'#eee8aa',
        'palegreen':'#98fb98',
        'paleturquoise':'#afeeee',
        'palevioletred':'#d87093',
        'papayawhip':'#ffefd5',
        'peachpuff':'#ffdab9',
        'peru':'#cd853f',
        'pink':'#ffc0cb',
        'plum':'#dda0dd',
        'powderblue':'#b0e0e6',
        'purple':'#800080',
        'red':'#ff0000',
        'rosybrown':'#bc8f8f',
        'royalblue':'#4169e1',
        'saddlebrown':'#8b4513',
        'salmon':'#fa8072',
        'sandybrown':'#f4a460',
        'seagreen':'#2e8b57',
        'seashell':'#fff5ee',
        'sienna':'#a0522d',
        'silver':'#c0c0c0',
        'skyblue':'#87ceeb',
        'slateblue':'#6a5acd',
        'slategray':'#708090',
        'slategrey':'#708090',
        'snow':'#fffafa',
        'springgreen':'#00ff7f',
        'steelblue':'#4682b4',
        'tan':'#d2b48c',
        'teal':'#008080',
        'thistle':'#d8bfd8',
        'tomato':'#ff6347',
        // 'transparent':'rgba(0,0,0,0)',
        'turquoise':'#40e0d0',
        'violet':'#ee82ee',
        'wheat':'#f5deb3',
        'white':'#ffffff',
        'whitesmoke':'#f5f5f5',
        'yellow':'#ffff00',
        'yellowgreen':'#9acd32'
    };
})(require('./tree'));
(function (tree) {

tree.Alpha = function (val) {
    this.value = val;
};
tree.Alpha.prototype = {
    toCSS: function () {
        return "alpha(opacity=" +
               (this.value.toCSS ? this.value.toCSS() : this.value) + ")";
    },
    eval: function (env) {
        if (this.value.eval) { this.value = this.value.eval(env) }
        return this;
    }
};

})(require('../tree'));
(function (tree) {

tree.Anonymous = function (string) {
    this.value = string.value || string;
};
tree.Anonymous.prototype = {
    toCSS: function () {
        return this.value;
    },
    eval: function () { return this },
    compare: function (x) {
        if (!x.toCSS) {
            return -1;
        }
        
        var left = this.toCSS(),
            right = x.toCSS();
        
        if (left === right) {
            return 0;
        }
        
        return left < right ? -1 : 1;
    }
};

})(require('../tree'));
(function (tree) {

tree.Assignment = function (key, val) {
    this.key = key;
    this.value = val;
};
tree.Assignment.prototype = {
    toCSS: function () {
        return this.key + '=' + (this.value.toCSS ? this.value.toCSS() : this.value);
    },
    eval: function (env) {
        if (this.value.eval) {
            return new(tree.Assignment)(this.key, this.value.eval(env));
        }
        return this;
    }
};

})(require('../tree'));(function (tree) {

//
// A function call node.
//
tree.Call = function (name, args, index, filename) {
    this.name = name;
    this.args = args;
    this.index = index;
    this.filename = filename;
};
tree.Call.prototype = {
    //
    // When evaluating a function call,
    // we either find the function in `tree.functions` [1],
    // in which case we call it, passing the  evaluated arguments,
    // or we simply print it out as it appeared originally [2].
    //
    // The *functions.js* file contains the built-in functions.
    //
    // The reason why we evaluate the arguments, is in the case where
    // we try to pass a variable to a function, like: `saturate(@color)`.
    // The function should receive the value, not the variable.
    //
    eval: function (env) {
        var args = this.args.map(function (a) { return a.eval(env) });

        if (this.name in tree.functions) { // 1.
            try {
                return tree.functions[this.name].apply(tree.functions, args);
            } catch (e) {
                throw { type: e.type || "Runtime",
                        message: "error evaluating function `" + this.name + "`" +
                                 (e.message ? ': ' + e.message : ''),
                        index: this.index, filename: this.filename };
            }
        } else { // 2.
            return new(tree.Anonymous)(this.name +
                   "(" + args.map(function (a) { return a.toCSS(env) }).join(', ') + ")");
        }
    },

    toCSS: function (env) {
        return this.eval(env).toCSS();
    }
};

})(require('../tree'));
(function (tree) {
//
// RGB Colors - #ff0014, #eee
//
tree.Color = function (rgb, a) {
    //
    // The end goal here, is to parse the arguments
    // into an integer triplet, such as `128, 255, 0`
    //
    // This facilitates operations and conversions.
    //
    if (Array.isArray(rgb)) {
        this.rgb = rgb;
    } else if (rgb.length == 6) {
        this.rgb = rgb.match(/.{2}/g).map(function (c) {
            return parseInt(c, 16);
        });
    } else {
        this.rgb = rgb.split('').map(function (c) {
            return parseInt(c + c, 16);
        });
    }
    this.alpha = typeof(a) === 'number' ? a : 1;
};
tree.Color.prototype = {
    eval: function () { return this },

    //
    // If we have some transparency, the only way to represent it
    // is via `rgba`. Otherwise, we use the hex representation,
    // which has better compatibility with older browsers.
    // Values are capped between `0` and `255`, rounded and zero-padded.
    //
    toCSS: function () {
        if (this.alpha < 1.0) {
            return "rgba(" + this.rgb.map(function (c) {
                return Math.round(c);
            }).concat(this.alpha).join(', ') + ")";
        } else {
            return '#' + this.rgb.map(function (i) {
                i = Math.round(i);
                i = (i > 255 ? 255 : (i < 0 ? 0 : i)).toString(16);
                return i.length === 1 ? '0' + i : i;
            }).join('');
        }
    },

    //
    // Operations have to be done per-channel, if not,
    // channels will spill onto each other. Once we have
    // our result, in the form of an integer triplet,
    // we create a new Color node to hold the result.
    //
    operate: function (op, other) {
        var result = [];

        if (! (other instanceof tree.Color)) {
            other = other.toColor();
        }

        for (var c = 0; c < 3; c++) {
            result[c] = tree.operate(op, this.rgb[c], other.rgb[c]);
        }
        return new(tree.Color)(result, this.alpha + other.alpha);
    },

    toHSL: function () {
        var r = this.rgb[0] / 255,
            g = this.rgb[1] / 255,
            b = this.rgb[2] / 255,
            a = this.alpha;

        var max = Math.max(r, g, b), min = Math.min(r, g, b);
        var h, s, l = (max + min) / 2, d = max - min;

        if (max === min) {
            h = s = 0;
        } else {
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2;               break;
                case b: h = (r - g) / d + 4;               break;
            }
            h /= 6;
        }
        return { h: h * 360, s: s, l: l, a: a };
    },
    toARGB: function () {
        var argb = [Math.round(this.alpha * 255)].concat(this.rgb);
        return '#' + argb.map(function (i) {
            i = Math.round(i);
            i = (i > 255 ? 255 : (i < 0 ? 0 : i)).toString(16);
            return i.length === 1 ? '0' + i : i;
        }).join('');
    },
    compare: function (x) {
        if (!x.rgb) {
            return -1;
        }
        
        return (x.rgb[0] === this.rgb[0] &&
            x.rgb[1] === this.rgb[1] &&
            x.rgb[2] === this.rgb[2] &&
            x.alpha === this.alpha) ? 0 : -1;
    }
};


})(require('../tree'));
(function (tree) {

tree.Comment = function (value, silent) {
    this.value = value;
    this.silent = !!silent;
};
tree.Comment.prototype = {
    toCSS: function (env) {
        return env.compress ? '' : this.value;
    },
    eval: function () { return this }
};

})(require('../tree'));
(function (tree) {

tree.Condition = function (op, l, r, i, negate) {
    this.op = op.trim();
    this.lvalue = l;
    this.rvalue = r;
    this.index = i;
    this.negate = negate;
};
tree.Condition.prototype.eval = function (env) {
    var a = this.lvalue.eval(env),
        b = this.rvalue.eval(env);

    var i = this.index, result;

    var result = (function (op) {
        switch (op) {
            case 'and':
                return a && b;
            case 'or':
                return a || b;
            default:
                if (a.compare) {
                    result = a.compare(b);
                } else if (b.compare) {
                    result = b.compare(a);
                } else {
                    throw { type: "Type",
                            message: "Unable to perform comparison",
                            index: i };
                }
                switch (result) {
                    case -1: return op === '<' || op === '=<';
                    case  0: return op === '=' || op === '>=' || op === '=<';
                    case  1: return op === '>' || op === '>=';
                }
        }
    })(this.op);
    return this.negate ? !result : result;
};

})(require('../tree'));
(function (tree) {

//
// A number with a unit
//
tree.Dimension = function (value, unit) {
    this.value = parseFloat(value);
    this.unit = unit || null;
};

tree.Dimension.prototype = {
    eval: function () { return this },
    toColor: function () {
        return new(tree.Color)([this.value, this.value, this.value]);
    },
    toCSS: function () {
        var css = this.value + this.unit;
        return css;
    },

    // In an operation between two Dimensions,
    // we default to the first Dimension's unit,
    // so `1px + 2em` will yield `3px`.
    // In the future, we could implement some unit
    // conversions such that `100cm + 10mm` would yield
    // `101cm`.
    operate: function (op, other) {
        return new(tree.Dimension)
                  (tree.operate(op, this.value, other.value),
                  this.unit || other.unit);
    },

    // TODO: Perform unit conversion before comparing
    compare: function (other) {
        if (other instanceof tree.Dimension) {
            if (other.value > this.value) {
                return -1;
            } else if (other.value < this.value) {
                return 1;
            } else {
                return 0;
            }
        } else {
            return -1;
        }
    }
};

})(require('../tree'));
(function (tree) {

tree.Directive = function (name, value) {
    this.name = name;

    if (Array.isArray(value)) {
        this.ruleset = new(tree.Ruleset)([], value);
        this.ruleset.allowImports = true;
    } else {
        this.value = value;
    }
};
tree.Directive.prototype = {
    toCSS: function (ctx, env) {
        if (this.ruleset) {
            this.ruleset.root = true;
            return this.name + (env.compress ? '{' : ' {\n  ') +
                   this.ruleset.toCSS(ctx, env).trim().replace(/\n/g, '\n  ') +
                               (env.compress ? '}': '\n}\n');
        } else {
            return this.name + ' ' + this.value.toCSS() + ';\n';
        }
    },
    eval: function (env) {
        var evaldDirective = this;
        if (this.ruleset) {
            env.frames.unshift(this);
            evaldDirective = new(tree.Directive)(this.name);
            evaldDirective.ruleset = this.ruleset.eval(env);
            env.frames.shift();
        }
        return evaldDirective;
    },
    variable: function (name) { return tree.Ruleset.prototype.variable.call(this.ruleset, name) },
    find: function () { return tree.Ruleset.prototype.find.apply(this.ruleset, arguments) },
    rulesets: function () { return tree.Ruleset.prototype.rulesets.apply(this.ruleset) }
};

})(require('../tree'));
(function (tree) {

tree.Element = function (combinator, value, index) {
    this.combinator = combinator instanceof tree.Combinator ?
                      combinator : new(tree.Combinator)(combinator);

    if (typeof(value) === 'string') {
        this.value = value.trim();
    } else if (value) {
        this.value = value;
    } else {
        this.value = "";
    }
    this.index = index;
};
tree.Element.prototype.eval = function (env) {
    return new(tree.Element)(this.combinator,
                             this.value.eval ? this.value.eval(env) : this.value,
                             this.index);
};
tree.Element.prototype.toCSS = function (env) {
	var value = (this.value.toCSS ? this.value.toCSS(env) : this.value);
	if (value == '' && this.combinator.value.charAt(0) == '&') {
		return '';
	} else {
		return this.combinator.toCSS(env || {}) + value;
	}
};

tree.Combinator = function (value) {
    if (value === ' ') {
        this.value = ' ';
    } else {
        this.value = value ? value.trim() : "";
    }
};
tree.Combinator.prototype.toCSS = function (env) {
    return {
        ''  : '',
        ' ' : ' ',
        ':' : ' :',
        '+' : env.compress ? '+' : ' + ',
        '~' : env.compress ? '~' : ' ~ ',
        '>' : env.compress ? '>' : ' > '
    }[this.value];
};

})(require('../tree'));
(function (tree) {

tree.Expression = function (value) { this.value = value };
tree.Expression.prototype = {
    eval: function (env) {
        if (this.value.length > 1) {
            return new(tree.Expression)(this.value.map(function (e) {
                return e.eval(env);
            }));
        } else if (this.value.length === 1) {
            return this.value[0].eval(env);
        } else {
            return this;
        }
    },
    toCSS: function (env) {
        return this.value.map(function (e) {
            return e.toCSS ? e.toCSS(env) : '';
        }).join(' ');
    }
};

})(require('../tree'));
(function (tree) {
//
// CSS @import node
//
// The general strategy here is that we don't want to wait
// for the parsing to be completed, before we start importing
// the file. That's because in the context of a browser,
// most of the time will be spent waiting for the server to respond.
//
// On creation, we push the import path to our import queue, though
// `import,push`, we also pass it a callback, which it'll call once
// the file has been fetched, and parsed.
//
tree.Import = function (path, imports, features, once, index) {
    var that = this;

    this.once = once;
    this.index = index;
    this._path = path;
    this.features = features && new(tree.Value)(features);

    // The '.less' extension is optional
    if (path instanceof tree.Quoted) {
        this.path = /\.(le?|c)ss(\?.*)?$/.test(path.value) ? path.value : path.value + '.less';
    } else {
        this.path = path.value.value || path.value;
    }

    this.css = /css(\?.*)?$/.test(this.path);

    // Only pre-compile .less files
    if (! this.css) {
        imports.push(this.path, function (e, root, imported) {
            if (e) { e.index = index }
            if (imported && that.once) that.skip = imported;
            that.root = root || new(tree.Ruleset)([], []);
        });
    }
};

//
// The actual import node doesn't return anything, when converted to CSS.
// The reason is that it's used at the evaluation stage, so that the rules
// it imports can be treated like any other rules.
//
// In `eval`, we make sure all Import nodes get evaluated, recursively, so
// we end up with a flat structure, which can easily be imported in the parent
// ruleset.
//
tree.Import.prototype = {
    toCSS: function (env) {
        var features = this.features ? ' ' + this.features.toCSS(env) : '';

        if (this.css) {
            return "@import " + this._path.toCSS() + features + ';\n';
        } else {
            return "";
        }
    },
    eval: function (env) {
        var ruleset, features = this.features && this.features.eval(env);

        if (this.skip) return [];

        if (this.css) {
            return this;
        } else {
            ruleset = new(tree.Ruleset)([], this.root.rules.slice(0));

            for (var i = 0; i < ruleset.rules.length; i++) {
                if (ruleset.rules[i] instanceof tree.Import) {
                    Array.prototype
                         .splice
                         .apply(ruleset.rules,
                                [i, 1].concat(ruleset.rules[i].eval(env)));
                }
            }
            return this.features ? new(tree.Media)(ruleset.rules, this.features.value) : ruleset.rules;
        }
    }
};

})(require('../tree'));
(function (tree) {

tree.JavaScript = function (string, index, escaped) {
    this.escaped = escaped;
    this.expression = string;
    this.index = index;
};
tree.JavaScript.prototype = {
    eval: function (env) {
        var result,
            that = this,
            context = {};

        var expression = this.expression.replace(/@\{([\w-]+)\}/g, function (_, name) {
            return tree.jsify(new(tree.Variable)('@' + name, that.index).eval(env));
        });

        try {
            expression = new(Function)('return (' + expression + ')');
        } catch (e) {
            throw { message: "JavaScript evaluation error: `" + expression + "`" ,
                    index: this.index };
        }

        for (var k in env.frames[0].variables()) {
            context[k.slice(1)] = {
                value: env.frames[0].variables()[k].value,
                toJS: function () {
                    return this.value.eval(env).toCSS();
                }
            };
        }

        try {
            result = expression.call(context);
        } catch (e) {
            throw { message: "JavaScript evaluation error: '" + e.name + ': ' + e.message + "'" ,
                    index: this.index };
        }
        if (typeof(result) === 'string') {
            return new(tree.Quoted)('"' + result + '"', result, this.escaped, this.index);
        } else if (Array.isArray(result)) {
            return new(tree.Anonymous)(result.join(', '));
        } else {
            return new(tree.Anonymous)(result);
        }
    }
};

})(require('../tree'));

(function (tree) {

tree.Keyword = function (value) { this.value = value };
tree.Keyword.prototype = {
    eval: function () { return this },
    toCSS: function () { return this.value },
    compare: function (other) {
        if (other instanceof tree.Keyword) {
            return other.value === this.value ? 0 : 1;
        } else {
            return -1;
        }
    }
};

tree.True = new(tree.Keyword)('true');
tree.False = new(tree.Keyword)('false');

})(require('../tree'));
(function (tree) {

tree.Media = function (value, features) {
    var selectors = this.emptySelectors();

    this.features = new(tree.Value)(features);
    this.ruleset = new(tree.Ruleset)(selectors, value);
    this.ruleset.allowImports = true;
};
tree.Media.prototype = {
    toCSS: function (ctx, env) {
        var features = this.features.toCSS(env);

        this.ruleset.root = (ctx.length === 0 || ctx[0].multiMedia);
        return '@media ' + features + (env.compress ? '{' : ' {\n  ') +
               this.ruleset.toCSS(ctx, env).trim().replace(/\n/g, '\n  ') +
                           (env.compress ? '}': '\n}\n');
    },
    eval: function (env) {
        if (!env.mediaBlocks) {
            env.mediaBlocks = [];
            env.mediaPath = [];
        }
        
        var blockIndex = env.mediaBlocks.length;
        env.mediaPath.push(this);
        env.mediaBlocks.push(this);

        var media = new(tree.Media)([], []);
        if(this.debugInfo) {
            this.ruleset.debugInfo = this.debugInfo;
            media.debugInfo = this.debugInfo;
        }
        media.features = this.features.eval(env);
        
        env.frames.unshift(this.ruleset);
        media.ruleset = this.ruleset.eval(env);
        env.frames.shift();
        
        env.mediaBlocks[blockIndex] = media;
        env.mediaPath.pop();

        return env.mediaPath.length === 0 ? media.evalTop(env) :
                    media.evalNested(env)
    },
    variable: function (name) { return tree.Ruleset.prototype.variable.call(this.ruleset, name) },
    find: function () { return tree.Ruleset.prototype.find.apply(this.ruleset, arguments) },
    rulesets: function () { return tree.Ruleset.prototype.rulesets.apply(this.ruleset) },
    emptySelectors: function() { 
        var el = new(tree.Element)('', '&', 0);
        return [new(tree.Selector)([el])];
    },

    evalTop: function (env) {
        var result = this;

        // Render all dependent Media blocks.
        if (env.mediaBlocks.length > 1) {
            var selectors = this.emptySelectors();
            result = new(tree.Ruleset)(selectors, env.mediaBlocks);
            result.multiMedia = true;
        }

        delete env.mediaBlocks;
        delete env.mediaPath;

        return result;
    },
    evalNested: function (env) {
        var i, value,
            path = env.mediaPath.concat([this]);

        // Extract the media-query conditions separated with `,` (OR).
        for (i = 0; i < path.length; i++) {
            value = path[i].features instanceof tree.Value ?
                        path[i].features.value : path[i].features;
            path[i] = Array.isArray(value) ? value : [value];
        }

        // Trace all permutations to generate the resulting media-query.
        //
        // (a, b and c) with nested (d, e) ->
        //    a and d
        //    a and e
        //    b and c and d
        //    b and c and e
        this.features = new(tree.Value)(this.permute(path).map(function (path) {
            path = path.map(function (fragment) {
                return fragment.toCSS ? fragment : new(tree.Anonymous)(fragment);
            });

            for(i = path.length - 1; i > 0; i--) {
                path.splice(i, 0, new(tree.Anonymous)("and"));
            }

            return new(tree.Expression)(path);
        }));

        // Fake a tree-node that doesn't output anything.
        return new(tree.Ruleset)([], []);
    },
    permute: function (arr) {
      if (arr.length === 0) {
          return [];
      } else if (arr.length === 1) {
          return arr[0];
      } else {
          var result = [];
          var rest = this.permute(arr.slice(1));
          for (var i = 0; i < rest.length; i++) {
              for (var j = 0; j < arr[0].length; j++) {
                  result.push([arr[0][j]].concat(rest[i]));
              }
          }
          return result;
      }
    },
    bubbleSelectors: function (selectors) {
      this.ruleset = new(tree.Ruleset)(selectors.slice(0), [this.ruleset]);
    }
};

})(require('../tree'));
(function (tree) {

tree.mixin = {};
tree.mixin.Call = function (elements, args, index, filename, important) {
    this.selector = new(tree.Selector)(elements);
    this.arguments = args;
    this.index = index;
    this.filename = filename;
    this.important = important;
};
tree.mixin.Call.prototype = {
    eval: function (env) {
        var mixins, args, rules = [], match = false;

        for (var i = 0; i < env.frames.length; i++) {
            if ((mixins = env.frames[i].find(this.selector)).length > 0) {
                args = this.arguments && this.arguments.map(function (a) {
                    return { name: a.name, value: a.value.eval(env) };
                });
                for (var m = 0; m < mixins.length; m++) {
                    if (mixins[m].match(args, env)) {
                        try {
                            Array.prototype.push.apply(
                                  rules, mixins[m].eval(env, this.arguments, this.important).rules);
                            match = true;
                        } catch (e) {
                            throw { message: e.message, index: this.index, filename: this.filename, stack: e.stack };
                        }
                    }
                }
                if (match) {
                    return rules;
                } else {
                    throw { type:    'Runtime',
                            message: 'No matching definition was found for `' +
                                      this.selector.toCSS().trim() + '('      +
                                      this.arguments.map(function (a) {
                                          return a.toCSS();
                                      }).join(', ') + ")`",
                            index:   this.index, filename: this.filename };
                }
            }
        }
        throw { type: 'Name',
                message: this.selector.toCSS().trim() + " is undefined",
                index: this.index, filename: this.filename };
    }
};

tree.mixin.Definition = function (name, params, rules, condition, variadic) {
    this.name = name;
    this.selectors = [new(tree.Selector)([new(tree.Element)(null, name)])];
    this.params = params;
    this.condition = condition;
    this.variadic = variadic;
    this.arity = params.length;
    this.rules = rules;
    this._lookups = {};
    this.required = params.reduce(function (count, p) {
        if (!p.name || (p.name && !p.value)) { return count + 1 }
        else                                 { return count }
    }, 0);
    this.parent = tree.Ruleset.prototype;
    this.frames = [];
};
tree.mixin.Definition.prototype = {
    toCSS:     function ()     { return "" },
    variable:  function (name) { return this.parent.variable.call(this, name) },
    variables: function ()     { return this.parent.variables.call(this) },
    find:      function ()     { return this.parent.find.apply(this, arguments) },
    rulesets:  function ()     { return this.parent.rulesets.apply(this) },

    evalParams: function (env, args) {
        var frame = new(tree.Ruleset)(null, []), varargs, arg;

        for (var i = 0, val, name; i < this.params.length; i++) {
            arg = args && args[i]

            if (arg && arg.name) {
                frame.rules.unshift(new(tree.Rule)(arg.name, arg.value.eval(env)));
                args.splice(i, 1);
                i--;
                continue;
            }
			
            if (name = this.params[i].name) {
                if (this.params[i].variadic && args) {
                    varargs = [];
                    for (var j = i; j < args.length; j++) {
                        varargs.push(args[j].value.eval(env));
                    }
                    frame.rules.unshift(new(tree.Rule)(name, new(tree.Expression)(varargs).eval(env)));
                } else if (val = (arg && arg.value) || this.params[i].value) {
                    frame.rules.unshift(new(tree.Rule)(name, val.eval(env)));
                } else {
                    throw { type: 'Runtime', message: "wrong number of arguments for " + this.name +
                            ' (' + args.length + ' for ' + this.arity + ')' };
                }
            }
        }
        return frame;
    },
    eval: function (env, args, important) {
        var frame = this.evalParams(env, args), context, _arguments = [], rules, start;

        for (var i = 0; i < Math.max(this.params.length, args && args.length); i++) {
            _arguments.push((args[i] && args[i].value) || this.params[i].value);
        }
        frame.rules.unshift(new(tree.Rule)('@arguments', new(tree.Expression)(_arguments).eval(env)));

        rules = important ?
            this.rules.map(function (r) {
                return new(tree.Rule)(r.name, r.value, '!important', r.index);
            }) : this.rules.slice(0);

        return new(tree.Ruleset)(null, rules).eval({
            frames: [this, frame].concat(this.frames, env.frames)
        });
    },
    match: function (args, env) {
        var argsLength = (args && args.length) || 0, len, frame;

        if (! this.variadic) {
            if (argsLength < this.required)                               { return false }
            if (argsLength > this.params.length)                          { return false }
            if ((this.required > 0) && (argsLength > this.params.length)) { return false }
        }

        if (this.condition && !this.condition.eval({
            frames: [this.evalParams(env, args)].concat(env.frames)
        }))                                                           { return false }

        len = Math.min(argsLength, this.arity);

        for (var i = 0; i < len; i++) {
            if (!this.params[i].name) {
                if (args[i].value.eval(env).toCSS() != this.params[i].value.eval(env).toCSS()) {
                    return false;
                }
            }
        }
        return true;
    }
};

})(require('../tree'));
(function (tree) {

tree.Operation = function (op, operands) {
    this.op = op.trim();
    this.operands = operands;
};
tree.Operation.prototype.eval = function (env) {
    var a = this.operands[0].eval(env),
        b = this.operands[1].eval(env),
        temp;

    if (a instanceof tree.Dimension && b instanceof tree.Color) {
        if (this.op === '*' || this.op === '+') {
            temp = b, b = a, a = temp;
        } else {
            throw { name: "OperationError",
                    message: "Can't substract or divide a color from a number" };
        }
    }
    return a.operate(this.op, b);
};

tree.operate = function (op, a, b) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return a / b;
    }
};

})(require('../tree'));

(function (tree) {

tree.Paren = function (node) {
    this.value = node;
};
tree.Paren.prototype = {
    toCSS: function (env) {
        return '(' + this.value.toCSS(env) + ')';
    },
    eval: function (env) {
        return new(tree.Paren)(this.value.eval(env));
    }
};

})(require('../tree'));
(function (tree) {

tree.Quoted = function (str, content, escaped, i) {
    this.escaped = escaped;
    this.value = content || '';
    this.quote = str.charAt(0);
    this.index = i;
};
tree.Quoted.prototype = {
    toCSS: function () {
        if (this.escaped) {
            return this.value;
        } else {
            return this.quote + this.value + this.quote;
        }
    },
    eval: function (env) {
        var that = this;
        var value = this.value.replace(/`([^`]+)`/g, function (_, exp) {
            return new(tree.JavaScript)(exp, that.index, true).eval(env).value;
        }).replace(/@\{([\w-]+)\}/g, function (_, name) {
            var v = new(tree.Variable)('@' + name, that.index).eval(env);
            return ('value' in v) ? v.value : v.toCSS();
        });
        return new(tree.Quoted)(this.quote + value + this.quote, value, this.escaped, this.index);
    },
    compare: function (x) {
        if (!x.toCSS) {
            return -1;
        }
        
        var left = this.toCSS(),
            right = x.toCSS();
        
        if (left === right) {
            return 0;
        }
        
        return left < right ? -1 : 1;
    }
};

})(require('../tree'));
(function (tree) {

tree.Ratio = function (value) {
    this.value = value;
};
tree.Ratio.prototype = {
    toCSS: function (env) {
        return this.value;
    },
    eval: function () { return this }
};

})(require('../tree'));
(function (tree) {

tree.Rule = function (name, value, important, index, inline) {
    this.name = name;
    this.value = (value instanceof tree.Value) ? value : new(tree.Value)([value]);
    this.important = important ? ' ' + important.trim() : '';
    this.index = index;
    this.inline = inline || false;

    if (name.charAt(0) === '@') {
        this.variable = true;
    } else { this.variable = false }
};
tree.Rule.prototype.toCSS = function (env) {
    if (this.variable) { return "" }
    else {
        return this.name + (env.compress ? ':' : ': ') +
               this.value.toCSS(env) +
               this.important + (this.inline ? "" : ";");
    }
};

tree.Rule.prototype.eval = function (context) {
    return new(tree.Rule)(this.name,
                          this.value.eval(context),
                          this.important,
                          this.index, this.inline);
};

tree.Shorthand = function (a, b) {
    this.a = a;
    this.b = b;
};

tree.Shorthand.prototype = {
    toCSS: function (env) {
        return this.a.toCSS(env) + "/" + this.b.toCSS(env);
    },
    eval: function () { return this }
};

})(require('../tree'));
(function (tree) {

tree.Ruleset = function (selectors, rules, strictImports) {
    this.selectors = selectors;
    this.rules = rules;
    this._lookups = {};
    this.strictImports = strictImports;
};
tree.Ruleset.prototype = {
    eval: function (env) {
        var selectors = this.selectors && this.selectors.map(function (s) { return s.eval(env) });
        var ruleset = new(tree.Ruleset)(selectors, this.rules.slice(0), this.strictImports);
        var rules = [];
        
        ruleset.root = this.root;
        ruleset.allowImports = this.allowImports;

        if(this.debugInfo) {
            ruleset.debugInfo = this.debugInfo;
        }

        // push the current ruleset to the frames stack
        env.frames.unshift(ruleset);

        // Evaluate imports
        if (ruleset.root || ruleset.allowImports || !ruleset.strictImports) {
            for (var i = 0; i < ruleset.rules.length; i++) {
                if (ruleset.rules[i] instanceof tree.Import) {
                    rules = rules.concat(ruleset.rules[i].eval(env));
                } else {
                    rules.push(ruleset.rules[i]);
                }
            }
            ruleset.rules = rules;
            rules = [];
        }

        // Store the frames around mixin definitions,
        // so they can be evaluated like closures when the time comes.
        for (var i = 0; i < ruleset.rules.length; i++) {
            if (ruleset.rules[i] instanceof tree.mixin.Definition) {
                ruleset.rules[i].frames = env.frames.slice(0);
            }
        }
        
        var mediaBlockCount = (env.mediaBlocks && env.mediaBlocks.length) || 0;

        // Evaluate mixin calls.
        for (var i = 0; i < ruleset.rules.length; i++) {
            if (ruleset.rules[i] instanceof tree.mixin.Call) {
                rules = rules.concat(ruleset.rules[i].eval(env));
            } else {
                rules.push(ruleset.rules[i]);
            }
        }
        ruleset.rules = rules;

        // Evaluate everything else
        for (var i = 0, rule; i < ruleset.rules.length; i++) {
            rule = ruleset.rules[i];

            if (! (rule instanceof tree.mixin.Definition)) {
                ruleset.rules[i] = rule.eval ? rule.eval(env) : rule;
            }
        }

        // Pop the stack
        env.frames.shift();
        
        if (env.mediaBlocks) {
            for(var i = mediaBlockCount; i < env.mediaBlocks.length; i++) {
                env.mediaBlocks[i].bubbleSelectors(selectors);
            }
        }

        return ruleset;
    },
    match: function (args) {
        return !args || args.length === 0;
    },
    variables: function () {
        if (this._variables) { return this._variables }
        else {
            return this._variables = this.rules.reduce(function (hash, r) {
                if (r instanceof tree.Rule && r.variable === true) {
                    hash[r.name] = r;
                }
                return hash;
            }, {});
        }
    },
    variable: function (name) {
        return this.variables()[name];
    },
    rulesets: function () {
        if (this._rulesets) { return this._rulesets }
        else {
            return this._rulesets = this.rules.filter(function (r) {
                return (r instanceof tree.Ruleset) || (r instanceof tree.mixin.Definition);
            });
        }
    },
    find: function (selector, self) {
        self = self || this;
        var rules = [], rule, match,
            key = selector.toCSS();

        if (key in this._lookups) { return this._lookups[key] }

        this.rulesets().forEach(function (rule) {
            if (rule !== self) {
                for (var j = 0; j < rule.selectors.length; j++) {
                    if (match = selector.match(rule.selectors[j])) {
                        if (selector.elements.length > rule.selectors[j].elements.length) {
                            Array.prototype.push.apply(rules, rule.find(
                                new(tree.Selector)(selector.elements.slice(1)), self));
                        } else {
                            rules.push(rule);
                        }
                        break;
                    }
                }
            }
        });
        return this._lookups[key] = rules;
    },
    //
    // Entry point for code generation
    //
    //     `context` holds an array of arrays.
    //
    toCSS: function (context, env) {
        var css = [],      // The CSS output
            rules = [],    // node.Rule instances
           _rules = [],    //
            rulesets = [], // node.Ruleset instances
            paths = [],    // Current selectors
            selector,      // The fully rendered selector
            debugInfo,     // Line number debugging
            rule;

        if (! this.root) {
            this.joinSelectors(paths, context, this.selectors);
        }

        // Compile rules and rulesets
        for (var i = 0; i < this.rules.length; i++) {
            rule = this.rules[i];

            if (rule.rules || (rule instanceof tree.Directive) || (rule instanceof tree.Media)) {
                rulesets.push(rule.toCSS(paths, env));
            } else if (rule instanceof tree.Comment) {
                if (!rule.silent) {
                    if (this.root) {
                        rulesets.push(rule.toCSS(env));
                    } else {
                        rules.push(rule.toCSS(env));
                    }
                }
            } else {
                if (rule.toCSS && !rule.variable) {
                    rules.push(rule.toCSS(env));
                } else if (rule.value && !rule.variable) {
                    rules.push(rule.value.toString());
                }
            }
        } 

        rulesets = rulesets.join('');

        // If this is the root node, we don't render
        // a selector, or {}.
        // Otherwise, only output if this ruleset has rules.
        if (this.root) {
            css.push(rules.join(env.compress ? '' : '\n'));
        } else {
            if (rules.length > 0) {
                debugInfo = tree.debugInfo(env, this);
                selector = paths.map(function (p) {
                    return p.map(function (s) {
                        return s.toCSS(env);
                    }).join('').trim();
                }).join(env.compress ? ',' : ',\n');

                // Remove duplicates
                for (var i = rules.length - 1; i >= 0; i--) {
                    if (_rules.indexOf(rules[i]) === -1) {
                        _rules.unshift(rules[i]);
                    }
                }
                rules = _rules;

                css.push(debugInfo + selector + 
                        (env.compress ? '{' : ' {\n  ') +
                        rules.join(env.compress ? '' : '\n  ') +
                        (env.compress ? '}' : '\n}\n'));
            }
        }
        css.push(rulesets);

        return css.join('')  + (env.compress ? '\n' : '');
    },

    joinSelectors: function (paths, context, selectors) {
        for (var s = 0; s < selectors.length; s++) {
            this.joinSelector(paths, context, selectors[s]);
        }
    },

    joinSelector: function (paths, context, selector) {

        var i, j, k, 
            hasParentSelector, newSelectors, el, sel, parentSel, 
            newSelectorPath, afterParentJoin, newJoinedSelector, 
            newJoinedSelectorEmpty, lastSelector, currentElements,
            selectorsMultiplied;
    
        for (i = 0; i < selector.elements.length; i++) {
            el = selector.elements[i];
            if (el.value === '&') {
                hasParentSelector = true;
            }
        }
    
        if (!hasParentSelector) {
            if (context.length > 0) {
                for(i = 0; i < context.length; i++) {
                    paths.push(context[i].concat(selector));
                }
            }
            else {
                paths.push([selector]);
            }
            return;
        }

        // The paths are [[Selector]]
        // The first list is a list of comma seperated selectors
        // The inner list is a list of inheritance seperated selectors
        // e.g.
        // .a, .b {
        //   .c {
        //   }
        // }
        // == [[.a] [.c]] [[.b] [.c]]
        //

        // the elements from the current selector so far
        currentElements = [];
        // the current list of new selectors to add to the path.
        // We will build it up. We initiate it with one empty selector as we "multiply" the new selectors
        // by the parents
        newSelectors = [[]];

        for (i = 0; i < selector.elements.length; i++) {
            el = selector.elements[i];
            // non parent reference elements just get added
            if (el.value !== "&") {
                currentElements.push(el);
            } else {
                // the new list of selectors to add
                selectorsMultiplied = [];

                // merge the current list of non parent selector elements
                // on to the current list of selectors to add
                if (currentElements.length > 0) {
                    this.mergeElementsOnToSelectors(currentElements, newSelectors);
                }

                // loop through our current selectors
                for(j = 0; j < newSelectors.length; j++) {
                    sel = newSelectors[j];
                    // if we don't have any parent paths, the & might be in a mixin so that it can be used
                    // whether there are parents or not
                    if (context.length == 0) {
                        // the combinator used on el should now be applied to the next element instead so that
                        // it is not lost
                        if (sel.length > 0) {
                            sel[0].elements = sel[0].elements.slice(0);
                            sel[0].elements.push(new(tree.Element)(el.combinator, '', 0)); //new Element(el.Combinator,  ""));
                        }
                        selectorsMultiplied.push(sel);
                    }
                    else {
                        // and the parent selectors
                        for(k = 0; k < context.length; k++) {
                            parentSel = context[k];
                            // We need to put the current selectors
                            // then join the last selector's elements on to the parents selectors

                            // our new selector path
                            newSelectorPath = [];
                            // selectors from the parent after the join
                            afterParentJoin = [];
                            newJoinedSelectorEmpty = true;

                            //construct the joined selector - if & is the first thing this will be empty,
                            // if not newJoinedSelector will be the last set of elements in the selector
                            if (sel.length > 0) {
                                newSelectorPath = sel.slice(0);
                                lastSelector = newSelectorPath.pop();
                                newJoinedSelector = new(tree.Selector)(lastSelector.elements.slice(0));
                                newJoinedSelectorEmpty = false;
                            }
                            else {
                                newJoinedSelector = new(tree.Selector)([]);
                            }

                            //put together the parent selectors after the join
                            if (parentSel.length > 1) {
                                afterParentJoin = afterParentJoin.concat(parentSel.slice(1));
                            }

                            if (parentSel.length > 0) {
                                newJoinedSelectorEmpty = false;

                                // join the elements so far with the first part of the parent
                                newJoinedSelector.elements.push(new(tree.Element)(el.combinator, parentSel[0].elements[0].value, 0));
                                newJoinedSelector.elements = newJoinedSelector.elements.concat(parentSel[0].elements.slice(1));
                            }

                            if (!newJoinedSelectorEmpty) {
                                // now add the joined selector
                                newSelectorPath.push(newJoinedSelector);
                            }

                            // and the rest of the parent
                            newSelectorPath = newSelectorPath.concat(afterParentJoin);

                            // add that to our new set of selectors
                            selectorsMultiplied.push(newSelectorPath);
                        }
                    }
                }

                // our new selectors has been multiplied, so reset the state
                newSelectors = selectorsMultiplied;
                currentElements = [];
            }
        }

        // if we have any elements left over (e.g. .a& .b == .b)
        // add them on to all the current selectors
        if (currentElements.length > 0) {
            this.mergeElementsOnToSelectors(currentElements, newSelectors);
        }

        for(i = 0; i < newSelectors.length; i++) {
            paths.push(newSelectors[i]);
        }
    },
    
    mergeElementsOnToSelectors: function(elements, selectors) {
        var i, sel;

        if (selectors.length == 0) {
            selectors.push([ new(tree.Selector)(elements) ]);
            return;
        }

        for(i = 0; i < selectors.length; i++) {
            sel = selectors[i];

            // if the previous thing in sel is a parent this needs to join on to it
            if (sel.length > 0) {
                sel[sel.length - 1] = new(tree.Selector)(sel[sel.length - 1].elements.concat(elements));
            }
            else {
                sel.push(new(tree.Selector)(elements));
            }
        }
    }
};
})(require('../tree'));
(function (tree) {

tree.Selector = function (elements) {
    this.elements = elements;
};
tree.Selector.prototype.match = function (other) {
    var len  = this.elements.length,
        olen = other.elements.length,
        max  = Math.min(len, olen);

    if (len < olen) {
        return false;
    } else {
        for (var i = 0; i < max; i++) {
            if (this.elements[i].value !== other.elements[i].value) {
                return false;
            }
        }
    }
    return true;
};
tree.Selector.prototype.eval = function (env) {
    return new(tree.Selector)(this.elements.map(function (e) {
        return e.eval(env);
    }));
};
tree.Selector.prototype.toCSS = function (env) {
    if (this._css) { return this._css }
    
    if (this.elements[0].combinator.value === "") {
        this._css = ' ';
    } else {
        this._css = '';
    }
    
    this._css += this.elements.map(function (e) {
        if (typeof(e) === 'string') {
            return ' ' + e.trim();
        } else {
            return e.toCSS(env);
        }
    }).join('');
    
    return this._css;
};

})(require('../tree'));
(function (tree) {

tree.URL = function (val, paths) {
    this.value = val;
    this.paths = paths;
};
tree.URL.prototype = {
    toCSS: function () {
        return "url(" + this.value.toCSS() + ")";
    },
    eval: function (ctx) {
        var val = this.value.eval(ctx);

        // Add the base path if the URL is relative and we are in the browser
        if (typeof window !== 'undefined' && typeof val.value === "string" && !/^(?:[a-z-]+:|\/)/.test(val.value) && this.paths.length > 0) {
            val.value = this.paths[0] + (val.value.charAt(0) === '/' ? val.value.slice(1) : val.value);
        }

        return new(tree.URL)(val, this.paths);
    }
};

})(require('../tree'));
(function (tree) {

tree.Value = function (value) {
    this.value = value;
    this.is = 'value';
};
tree.Value.prototype = {
    eval: function (env) {
        if (this.value.length === 1) {
            return this.value[0].eval(env);
        } else {
            return new(tree.Value)(this.value.map(function (v) {
                return v.eval(env);
            }));
        }
    },
    toCSS: function (env) {
        return this.value.map(function (e) {
            return e.toCSS(env);
        }).join(env.compress ? ',' : ', ');
    }
};

})(require('../tree'));
(function (tree) {

tree.Variable = function (name, index, file) { this.name = name, this.index = index, this.file = file };
tree.Variable.prototype = {
    eval: function (env) {
        var variable, v, name = this.name;

        if (name.indexOf('@@') == 0) {
            name = '@' + new(tree.Variable)(name.slice(1)).eval(env).value;
        }

        if (variable = tree.find(env.frames, function (frame) {
            if (v = frame.variable(name)) {
                return v.value.eval(env);
            }
        })) { return variable }
        else {
            throw { type: 'Name',
                    message: "variable " + name + " is undefined",
                    filename: this.file,
                    index: this.index };
        }
    }
};

})(require('../tree'));
(function (tree) {

tree.debugInfo = function(env, ctx) {
    var result="";
    if (env.dumpLineNumbers && !env.compress) {
        switch(env.dumpLineNumbers) {
            case 'comments':
                result = tree.debugInfo.asComment(ctx);
                break;
            case 'mediaquery':
                result = tree.debugInfo.asMediaQuery(ctx);
                break;
            case 'all':
                result = tree.debugInfo.asComment(ctx)+tree.debugInfo.asMediaQuery(ctx);
                break;
        }
    }
    return result;
};

tree.debugInfo.asComment = function(ctx) {
    return '/* line ' + ctx.debugInfo.lineNumber + ', ' + ctx.debugInfo.fileName + ' */\n';
};

tree.debugInfo.asMediaQuery = function(ctx) {
    return '@media -sass-debug-info{filename{font-family:"' + ctx.debugInfo.fileName + '";}line{font-family:"' + ctx.debugInfo.lineNumber + '";}}\n';
};

tree.find = function (obj, fun) {
    for (var i = 0, r; i < obj.length; i++) {
        if (r = fun.call(obj, obj[i])) { return r }
    }
    return null;
};
tree.jsify = function (obj) {
    if (Array.isArray(obj.value) && (obj.value.length > 1)) {
        return '[' + obj.value.map(function (v) { return v.toCSS(false) }).join(', ') + ']';
    } else {
        return obj.toCSS(false);
    }
};

})(require('./tree'));
//
// browser.js - client-side engine
//

var isFileProtocol = /^(file|chrome(-extension)?|resource|qrc|app):/.test(location.protocol);

less.env = less.env || (location.hostname == '127.0.0.1' ||
                        location.hostname == '0.0.0.0'   ||
                        location.hostname == 'localhost' ||
                        location.port.length > 0         ||
                        isFileProtocol                   ? 'development'
                                                         : 'production');

// Load styles asynchronously (default: false)
//
// This is set to `false` by default, so that the body
// doesn't start loading before the stylesheets are parsed.
// Setting this to `true` can result in flickering.
//
less.async = less.async || false;
less.fileAsync = less.fileAsync || false;

// Interval between watch polls
less.poll = less.poll || (isFileProtocol ? 1000 : 1500);

//
// Watch mode
//
less.watch   = function () { return this.watchMode = true };
less.unwatch = function () { return this.watchMode = false };

if (less.env === 'development') {
    less.optimization = 0;

    if (/!watch/.test(location.hash)) {
        less.watch();
    }
    var dumpLineNumbers = /!dumpLineNumbers:(comments|mediaquery|all)/.exec(location.hash);
    if (dumpLineNumbers) {
        less.dumpLineNumbers = dumpLineNumbers[1];
    }
    less.watchTimer = setInterval(function () {
        if (less.watchMode) {
            loadStyleSheets(function (e, root, _, sheet, env) {
                if (root) {
                    createCSS(root.toCSS(), sheet, env.lastModified);
                }
            });
        }
    }, less.poll);
} else {
    less.optimization = 3;
}

var cache;

try {
    cache = (typeof(window.localStorage) === 'undefined') ? null : window.localStorage;
} catch (_) {
    cache = null;
}

//
// Get all <link> tags with the 'rel' attribute set to "stylesheet/less"
//
var links = document.getElementsByTagName('link');
var typePattern = /^text\/(x-)?less$/;

less.sheets = [];

for (var i = 0; i < links.length; i++) {
    if (links[i].rel === 'stylesheet/less' || (links[i].rel.match(/stylesheet/) &&
       (links[i].type.match(typePattern)))) {
        less.sheets.push(links[i]);
    }
}


less.refresh = function (reload) {
    var startTime, endTime;
    startTime = endTime = new(Date);

    loadStyleSheets(function (e, root, _, sheet, env) {
        if (env.local) {
            log("loading " + sheet.href + " from cache.");
        } else {
            log("parsed " + sheet.href + " successfully.");
            createCSS(root.toCSS(), sheet, env.lastModified);
        }
        log("css for " + sheet.href + " generated in " + (new(Date) - endTime) + 'ms');
        (env.remaining === 0) && log("css generated in " + (new(Date) - startTime) + 'ms');
        endTime = new(Date);
    }, reload);

    loadStyles();
};
less.refreshStyles = loadStyles;

less.refresh(less.env === 'development');

function loadStyles() {
    var styles = document.getElementsByTagName('style');
    for (var i = 0; i < styles.length; i++) {
        if (styles[i].type.match(typePattern)) {
            new(less.Parser)({
                filename: document.location.href.replace(/#.*$/, ''),
                dumpLineNumbers: less.dumpLineNumbers
            }).parse(styles[i].innerHTML || '', function (e, tree) {
                var css = tree.toCSS();
                var style = styles[i];
                style.type = 'text/css';
                if (style.styleSheet) {
                    style.styleSheet.cssText = css;
                } else {
                    style.innerHTML = css;
                }
            });
        }
    }
}

function loadStyleSheets(callback, reload) {
    for (var i = 0; i < less.sheets.length; i++) {
        loadStyleSheet(less.sheets[i], callback, reload, less.sheets.length - (i + 1));
    }
}

function loadStyleSheet(sheet, callback, reload, remaining) {
    var contents  = sheet.contents || {};  // Passing a ref to top importing parser content cache trough 'sheet' arg.
    var url       = window.location.href.replace(/[#?].*$/, '');
    var href      = sheet.href.replace(/\?.*$/, '');
    var css       = cache && cache.getItem(href);
    var timestamp = cache && cache.getItem(href + ':timestamp');
    var styles    = { css: css, timestamp: timestamp };

    // Stylesheets in IE don't always return the full path
    if (! /^[a-z-]+:/.test(href)) {
        if (href.charAt(0) == "/") {
            href = window.location.protocol + "//" + window.location.host + href;
        } else {
            href = url.slice(0, url.lastIndexOf('/') + 1) + href;
        }
    }
    xhr(sheet.href, sheet.type, function (data, lastModified) {
        if (!reload && styles && lastModified &&
           (new(Date)(lastModified).valueOf() ===
            new(Date)(styles.timestamp).valueOf())) {
            // Use local copy
            createCSS(styles.css, sheet);
            callback(null, null, data, sheet, { local: true, remaining: remaining });
        } else {
            // Use remote copy (re-parse)
            try {
                contents[href] = data;  // Updating top importing parser content cache
                new(less.Parser)({
                    optimization: less.optimization,
                    paths: [href.replace(/[\w\.-]+$/, '')],
                    mime: sheet.type,
                    filename: href,
                    'contents': contents,    // Passing top importing parser content cache ref down.
                    dumpLineNumbers: less.dumpLineNumbers
                }).parse(data, function (e, root) {
                    if (e) { return error(e, href) }
                    try {
                        callback(e, root, data, sheet, { local: false, lastModified: lastModified, remaining: remaining });
                        removeNode(document.getElementById('less-error-message:' + extractId(href)));
                    } catch (e) {
                        error(e, href);
                    }
                });
            } catch (e) {
                error(e, href);
            }
        }
    }, function (status, url) {
        throw new(Error)("Couldn't load " + url + " (" + status + ")");
    });
}

function extractId(href) {
    return href.replace(/^[a-z]+:\/\/?[^\/]+/, '' )  // Remove protocol & domain
               .replace(/^\//,                 '' )  // Remove root /
               .replace(/\?.*$/,               '' )  // Remove query
               .replace(/\.[^\.\/]+$/,         '' )  // Remove file extension
               .replace(/[^\.\w-]+/g,          '-')  // Replace illegal characters
               .replace(/\./g,                 ':'); // Replace dots with colons(for valid id)
}

function createCSS(styles, sheet, lastModified) {
    var css;

    // Strip the query-string
    var href = sheet.href ? sheet.href.replace(/\?.*$/, '') : '';

    // If there is no title set, use the filename, minus the extension
    var id = 'less:' + (sheet.title || extractId(href));

    // If the stylesheet doesn't exist, create a new node
    if ((css = document.getElementById(id)) === null) {
        css = document.createElement('style');
        css.type = 'text/css';
        if( sheet.media ){ css.media = sheet.media; }
        css.id = id;
        var nextEl = sheet && sheet.nextSibling || null;
        document.getElementsByTagName('head')[0].insertBefore(css, nextEl);
    }

    if (css.styleSheet) { // IE
        try {
            css.styleSheet.cssText = styles;
        } catch (e) {
            throw new(Error)("Couldn't reassign styleSheet.cssText.");
        }
    } else {
        (function (node) {
            if (css.childNodes.length > 0) {
                if (css.firstChild.nodeValue !== node.nodeValue) {
                    css.replaceChild(node, css.firstChild);
                }
            } else {
                css.appendChild(node);
            }
        })(document.createTextNode(styles));
    }

    // Don't update the local store if the file wasn't modified
    if (lastModified && cache) {
        log('saving ' + href + ' to cache.');
        try {
            cache.setItem(href, styles);
            cache.setItem(href + ':timestamp', lastModified);
        } catch(e) {
            //TODO - could do with adding more robust error handling
            log('failed to save');
        }
    }
}

function xhr(url, type, callback, errback) {
    var xhr = getXMLHttpRequest();
    var async = isFileProtocol ? less.fileAsync : less.async;

    if (typeof(xhr.overrideMimeType) === 'function') {
        xhr.overrideMimeType('text/css');
    }
    xhr.open('GET', url, async);
    xhr.setRequestHeader('Accept', type || 'text/x-less, text/css; q=0.9, */*; q=0.5');
    xhr.send(null);

    if (isFileProtocol && !less.fileAsync) {
        if (xhr.status === 0 || (xhr.status >= 200 && xhr.status < 300)) {
            callback(xhr.responseText);
        } else {
            errback(xhr.status, url);
        }
    } else if (async) {
        xhr.onreadystatechange = function () {
            if (xhr.readyState == 4) {
                handleResponse(xhr, callback, errback);
            }
        };
    } else {
        handleResponse(xhr, callback, errback);
    }

    function handleResponse(xhr, callback, errback) {
        if (xhr.status >= 200 && xhr.status < 300) {
            callback(xhr.responseText,
                     xhr.getResponseHeader("Last-Modified"));
        } else if (typeof(errback) === 'function') {
            errback(xhr.status, url);
        }
    }
}

function getXMLHttpRequest() {
    if (window.XMLHttpRequest) {
        return new(XMLHttpRequest);
    } else {
        try {
            return new(ActiveXObject)("MSXML2.XMLHTTP.3.0");
        } catch (e) {
            log("browser doesn't support AJAX.");
            return null;
        }
    }
}

function removeNode(node) {
    return node && node.parentNode.removeChild(node);
}

function log(str) {
    if (less.env == 'development' && typeof(console) !== "undefined") { console.log('less: ' + str) }
}

function error(e, href) {
    var id = 'less-error-message:' + extractId(href);
    var template = '<li><label>{line}</label><pre class="{class}">{content}</pre></li>';
    var elem = document.createElement('div'), timer, content, error = [];
    var filename = e.filename || href;
    var filenameNoPath = filename.match(/([^\/]+)$/)[1];

    elem.id        = id;
    elem.className = "less-error-message";

    content = '<h3>'  + (e.message || 'There is an error in your .less file') +
              '</h3>' + '<p>in <a href="' + filename   + '">' + filenameNoPath + "</a> ";

    var errorline = function (e, i, classname) {
        if (e.extract[i]) {
            error.push(template.replace(/\{line\}/, parseInt(e.line) + (i - 1))
                               .replace(/\{class\}/, classname)
                               .replace(/\{content\}/, e.extract[i]));
        }
    };

    if (e.stack) {
        content += '<br/>' + e.stack.split('\n').slice(1).join('<br/>');
    } else if (e.extract) {
        errorline(e, 0, '');
        errorline(e, 1, 'line');
        errorline(e, 2, '');
        content += 'on line ' + e.line + ', column ' + (e.column + 1) + ':</p>' +
                    '<ul>' + error.join('') + '</ul>';
    }
    elem.innerHTML = content;

    // CSS for error messages
    createCSS([
        '.less-error-message ul, .less-error-message li {',
            'list-style-type: none;',
            'margin-right: 15px;',
            'padding: 4px 0;',
            'margin: 0;',
        '}',
        '.less-error-message label {',
            'font-size: 12px;',
            'margin-right: 15px;',
            'padding: 4px 0;',
            'color: #cc7777;',
        '}',
        '.less-error-message pre {',
            'color: #dd6666;',
            'padding: 4px 0;',
            'margin: 0;',
            'display: inline-block;',
        '}',
        '.less-error-message pre.line {',
            'color: #ff0000;',
        '}',
        '.less-error-message h3 {',
            'font-size: 20px;',
            'font-weight: bold;',
            'padding: 15px 0 5px 0;',
            'margin: 0;',
        '}',
        '.less-error-message a {',
            'color: #10a',
        '}',
        '.less-error-message .error {',
            'color: red;',
            'font-weight: bold;',
            'padding-bottom: 2px;',
            'border-bottom: 1px dashed red;',
        '}'
    ].join('\n'), { title: 'error-message' });

    elem.style.cssText = [
        "font-family: Arial, sans-serif",
        "border: 1px solid #e00",
        "background-color: #eee",
        "border-radius: 5px",
        "-webkit-border-radius: 5px",
        "-moz-border-radius: 5px",
        "color: #e00",
        "padding: 15px",
        "margin-bottom: 15px"
    ].join(';');

    if (less.env == 'development') {
        timer = setInterval(function () {
            if (document.body) {
                if (document.getElementById(id)) {
                    document.body.replaceChild(elem, document.getElementById(id));
                } else {
                    document.body.insertBefore(elem, document.body.firstChild);
                }
                clearInterval(timer);
            }
        }, 10);
    }
}

// amd.js
//
// Define Less as an AMD module.
if (typeof define === "function" && define.amd) {
    define("less", [], function () { return less; } );
}
})(window);
