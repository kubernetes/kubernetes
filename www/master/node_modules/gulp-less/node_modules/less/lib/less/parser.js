var less, tree;

// Node.js does not have a header file added which defines less
if (less === undefined) {
    less = exports;
    tree = require('./tree');
    less.mode = 'node';
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
//      `j` holds the current chunk index, and `currentPos` holds
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
        saveStack = [],   // holds state for backtracking
        furthest,    // furthest index the parser has gone to
        chunks,      // chunkified input
        current,     // current chunk
        currentPos,  // index of current chunk, in `input`
        parser,
        parsers,
        rootFilename = env && env.filename;

    // Top parser on an import tree must be sure there is one "env"
    // which will then be passed around by reference.
    if (!(env instanceof tree.parseEnv)) {
        env = new tree.parseEnv(env);
    }

    var imports = this.imports = {
        paths: env.paths || [],  // Search paths, when importing
        queue: [],               // Files which haven't been imported yet
        files: env.files,        // Holds the imported parse trees
        contents: env.contents,  // Holds the imported file contents
        contentsIgnoredChars: env.contentsIgnoredChars, // lines inserted, not in the original less
        mime:  env.mime,         // MIME type of .less files
        error: null,             // Error in parsing/evaluating an import
        push: function (path, currentFileInfo, importOptions, callback) {
            var parserImports = this;
            this.queue.push(path);

            var fileParsedFunc = function (e, root, fullPath) {
                parserImports.queue.splice(parserImports.queue.indexOf(path), 1); // Remove the path from the queue

                var importedPreviously = fullPath === rootFilename;

                parserImports.files[fullPath] = root;                        // Store the root

                if (e && !parserImports.error) { parserImports.error = e; }

                callback(e, root, importedPreviously, fullPath);
            };

            if (less.Parser.importer) {
                less.Parser.importer(path, currentFileInfo, fileParsedFunc, env);
            } else {
                less.Parser.fileLoader(path, currentFileInfo, function(e, contents, fullPath, newFileInfo) {
                    if (e) {fileParsedFunc(e); return;}

                    var newEnv = new tree.parseEnv(env);

                    newEnv.currentFileInfo = newFileInfo;
                    newEnv.processImports = false;
                    newEnv.contents[fullPath] = contents;

                    if (currentFileInfo.reference || importOptions.reference) {
                        newFileInfo.reference = true;
                    }

                    if (importOptions.inline) {
                        fileParsedFunc(null, contents, fullPath);
                    } else {
                        new(less.Parser)(newEnv).parse(contents, function (e, root) {
                            fileParsedFunc(e, root, fullPath);
                        });
                    }
                }, env);
            }
        }
    };

    function save()    { currentPos = i; saveStack.push( { current: current, i: i, j: j }); }
    function restore() { var state = saveStack.pop(); current = state.current; currentPos = i = state.i; j = state.j; }
    function forget() { saveStack.pop(); }

    function sync() {
        if (i > currentPos) {
            current = current.slice(i - currentPos);
            currentPos = i;
        }
    }
    function isWhitespace(str, pos) {
        var code = str.charCodeAt(pos | 0);
        return (code <= 32) && (code === 32 || code === 10 || code === 9);
    }
    //
    // Parse from a token, regexp or string, and move forward if match
    //
    function $(tok) {
        var tokType = typeof tok,
            match, length;

        // Either match a single character in the input,
        // or match a regexp in the current chunk (`current`).
        //
        if (tokType === "string") {
            if (input.charAt(i) !== tok) {
                return null;
            }
            skipWhitespace(1);
            return tok;
        }

        // regexp
        sync ();
        if (! (match = tok.exec(current))) {
            return null;
        }

        length = match[0].length;

        // The match is confirmed, add the match length to `i`,
        // and consume any extra white-space characters (' ' || '\n')
        // which come after that. The reason for this is that LeSS's
        // grammar is mostly white-space insensitive.
        //
        skipWhitespace(length);

        if(typeof(match) === 'string') {
            return match;
        } else {
            return match.length === 1 ? match[0] : match;
        }
    }

    // Specialization of $(tok)
    function $re(tok) {
        if (i > currentPos) {
            current = current.slice(i - currentPos);
            currentPos = i;
        }
        var m = tok.exec(current);
        if (!m) {
            return null;
        }

        skipWhitespace(m[0].length);
        if(typeof m === "string") {
            return m;
        }

        return m.length === 1 ? m[0] : m;
    }

    var _$re = $re;

    // Specialization of $(tok)
    function $char(tok) {
        if (input.charAt(i) !== tok) {
            return null;
        }
        skipWhitespace(1);
        return tok;
    }

    function skipWhitespace(length) {
        var oldi = i, oldj = j,
            curr = i - currentPos,
            endIndex = i + current.length - curr,
            mem = (i += length),
            inp = input,
            c;

        for (; i < endIndex; i++) {
            c = inp.charCodeAt(i);
            if (c > 32) {
                break;
            }

            if ((c !== 32) && (c !== 10) && (c !== 9) && (c !== 13)) {
                break;
            }
         }

        current = current.slice(length + i - mem + curr);
        currentPos = i;

        if (!current.length && (j < chunks.length - 1)) {
            current = chunks[++j];
            skipWhitespace(0); // skip space at the beginning of a chunk
            return true; // things changed
        }

        return oldi !== i || oldj !== j;
    }

    function expect(arg, msg, index) {
        // some older browsers return typeof 'function' for RegExp
        var result = (Object.prototype.toString.call(arg) === '[object Function]') ? arg.call(parsers) : $(arg);
        if (result) {
            return result;
        }
        error(msg || (typeof(arg) === 'string' ? "expected '" + arg + "' got '" + input.charAt(i) + "'"
                                               : "unexpected token"));
    }

    // Specialization of expect()
    function expectChar(arg, msg) {
        if (input.charAt(i) === arg) {
            skipWhitespace(1);
            return arg;
        }
        error(msg || "expected '" + arg + "' got '" + input.charAt(i) + "'");
    }

    function error(msg, type) {
        var e = new Error(msg);
        e.index = i;
        e.type = type || 'Syntax';
        throw e;
    }

    // Same as $(), but don't change the state of the parser,
    // just return the match.
    function peek(tok) {
        if (typeof(tok) === 'string') {
            return input.charAt(i) === tok;
        } else {
            return tok.test(current);
        }
    }

    // Specialization of peek()
    function peekChar(tok) {
        return input.charAt(i) === tok;
    }


    function getInput(e, env) {
        if (e.filename && env.currentFileInfo.filename && (e.filename !== env.currentFileInfo.filename)) {
            return parser.imports.contents[e.filename];
        } else {
            return input;
        }
    }

    function getLocation(index, inputStream) {
        var n = index + 1,
            line = null,
            column = -1;

        while (--n >= 0 && inputStream.charAt(n) !== '\n') {
            column++;
        }

        if (typeof index === 'number') {
            line = (inputStream.slice(0, index).match(/\n/g) || "").length;
        }

        return {
            line: line,
            column: column
        };
    }

    function getDebugInfo(index, inputStream, env) {
        var filename = env.currentFileInfo.filename;
        if(less.mode !== 'browser' && less.mode !== 'rhino') {
            filename = require('path').resolve(filename);
        }

        return {
            lineNumber: getLocation(index, inputStream).line + 1,
            fileName: filename
        };
    }

    function LessError(e, env) {
        var input = getInput(e, env),
            loc = getLocation(e.index, input),
            line = loc.line,
            col  = loc.column,
            callLine = e.call && getLocation(e.call, input).line,
            lines = input.split('\n');

        this.type = e.type || 'Syntax';
        this.message = e.message;
        this.filename = e.filename || env.currentFileInfo.filename;
        this.index = e.index;
        this.line = typeof(line) === 'number' ? line + 1 : null;
        this.callLine = callLine + 1;
        this.callExtract = lines[callLine];
        this.stack = e.stack;
        this.column = col;
        this.extract = [
            lines[line - 1],
            lines[line],
            lines[line + 1]
        ];
    }

    LessError.prototype = new Error();
    LessError.prototype.constructor = LessError;

    this.env = env = env || {};

    // The optimization level dictates the thoroughness of the parser,
    // the lower the number, the less nodes it will create in the tree.
    // This could matter for debugging, or if you want to access
    // the individual nodes in the tree.
    this.optimization = ('optimization' in this.env) ? this.env.optimization : 1;

    //
    // The Parser
    //
    parser = {

        imports: imports,
        //
        // Parse an input string into an abstract syntax tree,
        // @param str A string containing 'less' markup
        // @param callback call `callback` when done.
        // @param [additionalData] An optional map which can contains vars - a map (key, value) of variables to apply
        //
        parse: function (str, callback, additionalData) {
            var root, line, lines, error = null, globalVars, modifyVars, preText = "";

            i = j = currentPos = furthest = 0;

            globalVars = (additionalData && additionalData.globalVars) ? less.Parser.serializeVars(additionalData.globalVars) + '\n' : '';
            modifyVars = (additionalData && additionalData.modifyVars) ? '\n' + less.Parser.serializeVars(additionalData.modifyVars) : '';

            if (globalVars || (additionalData && additionalData.banner)) {
                preText = ((additionalData && additionalData.banner) ? additionalData.banner : "") + globalVars;
                parser.imports.contentsIgnoredChars[env.currentFileInfo.filename] = preText.length;
            }

            str = str.replace(/\r\n/g, '\n');
            // Remove potential UTF Byte Order Mark
            input = str = preText + str.replace(/^\uFEFF/, '') + modifyVars;
            parser.imports.contents[env.currentFileInfo.filename] = str;

            // Split the input into chunks.
            chunks = (function (input) {
                var len = input.length, level = 0, parenLevel = 0,
                    lastOpening, lastOpeningParen, lastMultiComment, lastMultiCommentEndBrace,
                    chunks = [], emitFrom = 0,
                    parserCurrentIndex, currentChunkStartIndex, cc, cc2, matched;

                function fail(msg, index) {
                    error = new(LessError)({
                        index: index || parserCurrentIndex,
                        type: 'Parse',
                        message: msg,
                        filename: env.currentFileInfo.filename
                    }, env);
                }

                function emitChunk(force) {
                    var len = parserCurrentIndex - emitFrom;
                    if (((len < 512) && !force) || !len) {
                        return;
                    }
                    chunks.push(input.slice(emitFrom, parserCurrentIndex + 1));
                    emitFrom = parserCurrentIndex + 1;
                }

                for (parserCurrentIndex = 0; parserCurrentIndex < len; parserCurrentIndex++) {
                    cc = input.charCodeAt(parserCurrentIndex);
                    if (((cc >= 97) && (cc <= 122)) || (cc < 34)) {
                        // a-z or whitespace
                        continue;
                    }

                    switch (cc) {
                        case 40:                        // (
                            parenLevel++;
                            lastOpeningParen = parserCurrentIndex;
                            continue;
                        case 41:                        // )
                            if (--parenLevel < 0) {
                                return fail("missing opening `(`");
                            }
                            continue;
                        case 59:                        // ;
                            if (!parenLevel) { emitChunk(); }
                            continue;
                        case 123:                       // {
                            level++;
                            lastOpening = parserCurrentIndex;
                            continue;
                        case 125:                       // }
                            if (--level < 0) {
                                return fail("missing opening `{`");
                            }
                            if (!level && !parenLevel) { emitChunk(); }
                            continue;
                        case 92:                        // \
                            if (parserCurrentIndex < len - 1) { parserCurrentIndex++; continue; }
                            return fail("unescaped `\\`");
                        case 34:
                        case 39:
                        case 96:                        // ", ' and `
                            matched = 0;
                            currentChunkStartIndex = parserCurrentIndex;
                            for (parserCurrentIndex = parserCurrentIndex + 1; parserCurrentIndex < len; parserCurrentIndex++) {
                                cc2 = input.charCodeAt(parserCurrentIndex);
                                if (cc2 > 96) { continue; }
                                if (cc2 == cc) { matched = 1; break; }
                                if (cc2 == 92) {        // \
                                    if (parserCurrentIndex == len - 1) {
                                        return fail("unescaped `\\`");
                                    }
                                    parserCurrentIndex++;
                                }
                            }
                            if (matched) { continue; }
                            return fail("unmatched `" + String.fromCharCode(cc) + "`", currentChunkStartIndex);
                        case 47:                        // /, check for comment
                            if (parenLevel || (parserCurrentIndex == len - 1)) { continue; }
                            cc2 = input.charCodeAt(parserCurrentIndex + 1);
                            if (cc2 == 47) {
                                // //, find lnfeed
                                for (parserCurrentIndex = parserCurrentIndex + 2; parserCurrentIndex < len; parserCurrentIndex++) {
                                    cc2 = input.charCodeAt(parserCurrentIndex);
                                    if ((cc2 <= 13) && ((cc2 == 10) || (cc2 == 13))) { break; }
                                }
                            } else if (cc2 == 42) {
                                // /*, find */
                                lastMultiComment = currentChunkStartIndex = parserCurrentIndex;
                                for (parserCurrentIndex = parserCurrentIndex + 2; parserCurrentIndex < len - 1; parserCurrentIndex++) {
                                    cc2 = input.charCodeAt(parserCurrentIndex);
                                    if (cc2 == 125) { lastMultiCommentEndBrace = parserCurrentIndex; }
                                    if (cc2 != 42) { continue; }
                                    if (input.charCodeAt(parserCurrentIndex + 1) == 47) { break; }
                                }
                                if (parserCurrentIndex == len - 1) {
                                    return fail("missing closing `*/`", currentChunkStartIndex);
                                }
                                parserCurrentIndex++;
                            }
                            continue;
                        case 42:                       // *, check for unmatched */
                            if ((parserCurrentIndex < len - 1) && (input.charCodeAt(parserCurrentIndex + 1) == 47)) {
                                return fail("unmatched `/*`");
                            }
                            continue;
                    }
                }

                if (level !== 0) {
                    if ((lastMultiComment > lastOpening) && (lastMultiCommentEndBrace > lastMultiComment)) {
                        return fail("missing closing `}` or `*/`", lastOpening);
                    } else {
                        return fail("missing closing `}`", lastOpening);
                    }
                } else if (parenLevel !== 0) {
                    return fail("missing closing `)`", lastOpeningParen);
                }

                emitChunk(true);
                return chunks;
            })(str);

            if (error) {
                return callback(new(LessError)(error, env));
            }

            current = chunks[0];

            // Start with the primary rule.
            // The whole syntax tree is held under a Ruleset node,
            // with the `root` property set to true, so no `{}` are
            // output. The callback is called when the input is parsed.
            try {
                root = new(tree.Ruleset)(null, this.parsers.primary());
                root.root = true;
                root.firstRoot = true;
            } catch (e) {
                return callback(new(LessError)(e, env));
            }

            root.toCSS = (function (evaluate) {
                return function (options, variables) {
                    options = options || {};
                    var evaldRoot,
                        css,
                        evalEnv = new tree.evalEnv(options);

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
                            return new(tree.Rule)('@' + k, value, false, null, 0);
                        });
                        evalEnv.frames = [new(tree.Ruleset)(null, variables)];
                    }

                    try {
                        var preEvalVisitors = [],
                            visitors = [
                                new(tree.joinSelectorVisitor)(),
                                new(tree.processExtendsVisitor)(),
                                new(tree.toCSSVisitor)({compress: Boolean(options.compress)})
                            ], i, root = this;

                        if (options.plugins) {
                            for(i =0; i < options.plugins.length; i++) {
                                if (options.plugins[i].isPreEvalVisitor) {
                                    preEvalVisitors.push(options.plugins[i]);
                                } else {
                                    if (options.plugins[i].isPreVisitor) {
                                        visitors.splice(0, 0, options.plugins[i]);
                                    } else {
                                        visitors.push(options.plugins[i]);
                                    }
                                }
                            }
                        }

                        for(i = 0; i < preEvalVisitors.length; i++) {
                            preEvalVisitors[i].run(root);
                        }

                        evaldRoot = evaluate.call(root, evalEnv);

                        for(i = 0; i < visitors.length; i++) {
                            visitors[i].run(evaldRoot);
                        }

                        if (options.sourceMap) {
                            evaldRoot = new tree.sourceMapOutput(
                                {
                                    contentsIgnoredCharsMap: parser.imports.contentsIgnoredChars,
                                    writeSourceMap: options.writeSourceMap,
                                    rootNode: evaldRoot,
                                    contentsMap: parser.imports.contents,
                                    sourceMapFilename: options.sourceMapFilename,
                                    sourceMapURL: options.sourceMapURL,
                                    outputFilename: options.sourceMapOutputFilename,
                                    sourceMapBasepath: options.sourceMapBasepath,
                                    sourceMapRootpath: options.sourceMapRootpath,
                                    outputSourceFiles: options.outputSourceFiles,
                                    sourceMapGenerator: options.sourceMapGenerator
                                });
                        }

                        css = evaldRoot.toCSS({
                                compress: Boolean(options.compress),
                                dumpLineNumbers: env.dumpLineNumbers,
                                strictUnits: Boolean(options.strictUnits),
                                numPrecision: 8});
                    } catch (e) {
                        throw new(LessError)(e, env);
                    }

                    if (options.cleancss && less.mode === 'node') {
                        var CleanCSS = require('clean-css'),
                            cleancssOptions = options.cleancssOptions || {};

                        if (cleancssOptions.keepSpecialComments === undefined) {
                            cleancssOptions.keepSpecialComments = "*";
                        }
                        cleancssOptions.processImport = false;
                        cleancssOptions.noRebase = true;
                        if (cleancssOptions.noAdvanced === undefined) {
                            cleancssOptions.noAdvanced = true;
                        }

                        return new CleanCSS(cleancssOptions).minify(css);
                    } else if (options.compress) {
                        return css.replace(/(^(\s)+)|((\s)+$)/g, "");
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
                var loc = getLocation(i, input);
                lines = input.split('\n');
                line = loc.line + 1;

                error = {
                    type: "Parse",
                    message: "Unrecognised input",
                    index: i,
                    filename: env.currentFileInfo.filename,
                    line: line,
                    column: loc.column,
                    extract: [
                        lines[line - 2],
                        lines[line - 1],
                        lines[line]
                    ]
                };
            }

            var finish = function (e) {
                e = error || e || parser.imports.error;

                if (e) {
                    if (!(e instanceof LessError)) {
                        e = new(LessError)(e, env);
                    }

                    return callback(e);
                }
                else {
                    return callback(null, root);
                }
            };

            if (env.processImports !== false) {
                new tree.importVisitor(this.imports, finish)
                    .run(root);
            } else {
                return finish();
            }
        },

        //
        // Here in, the parsing rules/functions
        //
        // The basic structure of the syntax tree generated is as follows:
        //
        //   Ruleset ->  Rule -> Value -> Expression -> Entity
        //
        // Here's some Less code:
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
        parsers: parsers = {
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
                var mixin = this.mixin, $re = _$re, root = [], node;

                while (current)
                {
                    node = this.extendRule() || mixin.definition() || this.rule() || this.ruleset() ||
                        mixin.call() || this.comment() || this.rulesetCall() || this.directive();
                    if (node) {
                        root.push(node);
                    } else {
                        if (!($re(/^[\s\n]+/) || $re(/^;+/))) {
                            break;
                        }
                    }
                    if (peekChar('}')) {
                        break;
                    }
                }

                return root;
            },

            // We create a Comment node for CSS comments `/* */`,
            // but keep the LeSS comments `//` silent, by just skipping
            // over them.
            comment: function () {
                var comment;

                if (input.charAt(i) !== '/') { return; }

                if (input.charAt(i + 1) === '/') {
                    return new(tree.Comment)($re(/^\/\/.*/), true, i, env.currentFileInfo);
                }
                comment = $re(/^\/\*(?:[^*]|\*+[^\/*])*\*+\/\n?/);
                if (comment) {
                    return new(tree.Comment)(comment, false, i, env.currentFileInfo);
                }
            },

            comments: function () {
                var comment, comments = [];

                while(true) {
                    comment = this.comment();
                    if (!comment) {
                        break;
                    }
                    comments.push(comment);
                }

                return comments;
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
                    var str, j = i, e, index = i;

                    if (input.charAt(j) === '~') { j++; e = true; } // Escaped strings
                    if (input.charAt(j) !== '"' && input.charAt(j) !== "'") { return; }

                    if (e) { $char('~'); }

                    str = $re(/^"((?:[^"\\\r\n]|\\.)*)"|'((?:[^'\\\r\n]|\\.)*)'/);
                    if (str) {
                        return new(tree.Quoted)(str[0], str[1] || str[2], e, index, env.currentFileInfo);
                    }
                },

                //
                // A catch-all word, such as:
                //
                //     black border-collapse
                //
                keyword: function () {
                    var k;

                    k = $re(/^%|^[_A-Za-z-][_A-Za-z0-9-]*/);
                    if (k) {
                        var color = tree.Color.fromKeyword(k);
                        if (color) {
                            return color;
                        }
                        return new(tree.Keyword)(k);
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

                    name = /^([\w-]+|%|progid:[\w\.]+)\(/.exec(current);
                    if (!name) { return; }

                    name = name[1];
                    nameLC = name.toLowerCase();
                    if (nameLC === 'url') {
                        return null;
                    }

                    i += name.length;

                    if (nameLC === 'alpha') {
                        alpha_ret = parsers.alpha();
                        if(typeof alpha_ret !== 'undefined') {
                            return alpha_ret;
                        }
                    }

                    $char('('); // Parse the '(' and consume whitespace.

                    args = this.arguments();

                    if (! $char(')')) {
                        return;
                    }

                    if (name) { return new(tree.Call)(name, args, index, env.currentFileInfo); }
                },
                arguments: function () {
                    var args = [], arg;

                    while (true) {
                        arg = this.assignment() || parsers.expression();
                        if (!arg) {
                            break;
                        }
                        args.push(arg);
                        if (! $char(',')) {
                            break;
                        }
                    }
                    return args;
                },
                literal: function () {
                    return this.dimension() ||
                           this.color() ||
                           this.quoted() ||
                           this.unicodeDescriptor();
                },

                // Assignments are argument entities for calls.
                // They are present in ie filter properties as shown below.
                //
                //     filter: progid:DXImageTransform.Microsoft.Alpha( *opacity=50* )
                //

                assignment: function () {
                    var key, value;
                    key = $re(/^\w+(?=\s?=)/i);
                    if (!key) {
                        return;
                    }
                    if (!$char('=')) {
                        return;
                    }
                    value = parsers.entity();
                    if (value) {
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

                    if (input.charAt(i) !== 'u' || !$re(/^url\(/)) {
                        return;
                    }

                    value = this.quoted() || this.variable() ||
                            $re(/^(?:(?:\\[\(\)'"])|[^\(\)'"])+/) || "";

                    expectChar(')');

                    return new(tree.URL)((value.value != null || value instanceof tree.Variable)
                                        ? value : new(tree.Anonymous)(value), env.currentFileInfo);
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

                    if (input.charAt(i) === '@' && (name = $re(/^@@?[\w-]+/))) {
                        return new(tree.Variable)(name, index, env.currentFileInfo);
                    }
                },

                // A variable entity useing the protective {} e.g. @{var}
                variableCurly: function () {
                    var curly, index = i;

                    if (input.charAt(i) === '@' && (curly = $re(/^@\{([\w-]+)\}/))) {
                        return new(tree.Variable)("@" + curly[1], index, env.currentFileInfo);
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

                    if (input.charAt(i) === '#' && (rgb = $re(/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})/))) {
                        var colorCandidateString = rgb.input.match(/^#([\w]+).*/); // strip colons, brackets, whitespaces and other characters that should not definitely be part of color string
                        colorCandidateString = colorCandidateString[1];
                        if (!colorCandidateString.match(/^[A-Fa-f0-9]+$/)) { // verify if candidate consists only of allowed HEX characters
                            error("Invalid HEX color code");
                        }
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
                    //Is the first char of the dimension 0-9, '.', '+' or '-'
                    if ((c > 57 || c < 43) || c === 47 || c == 44) {
                        return;
                    }

                    value = $re(/^([+-]?\d*\.?\d+)(%|[a-z]+)?/);
                    if (value) {
                        return new(tree.Dimension)(value[1], value[2]);
                    }
                },

                //
                // A unicode descriptor, as is used in unicode-range
                //
                // U+0??  or U+00A1-00A9
                //
                unicodeDescriptor: function () {
                    var ud;

                    ud = $re(/^U\+[0-9a-fA-F?]+(\-[0-9a-fA-F?]+)?/);
                    if (ud) {
                        return new(tree.UnicodeDescriptor)(ud[0]);
                    }
                },

                //
                // JavaScript code to be evaluated
                //
                //     `window.location.href`
                //
                javascript: function () {
                    var str, j = i, e;

                    if (input.charAt(j) === '~') { j++; e = true; } // Escaped strings
                    if (input.charAt(j) !== '`') { return; }
                    if (env.javascriptEnabled !== undefined && !env.javascriptEnabled) {
                        error("You are using JavaScript, which has been disabled.");
                    }

                    if (e) { $char('~'); }

                    str = $re(/^`([^`]*)`/);
                    if (str) {
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

                if (input.charAt(i) === '@' && (name = $re(/^(@[\w-]+)\s*:/))) { return name[1]; }
            },

            //
            // The variable part of a variable definition. Used in the `rule` parser
            //
            //     @fink();
            //
            rulesetCall: function () {
                var name;

                if (input.charAt(i) === '@' && (name = $re(/^(@[\w-]+)\s*\(\s*\)\s*;/))) {
                    return new tree.RulesetCall(name[1]);
                }
            },

            //
            // extend syntax - used to extend selectors
            //
            extend: function(isRule) {
                var elements, e, index = i, option, extendList, extend;

                if (!(isRule ? $re(/^&:extend\(/) : $re(/^:extend\(/))) { return; }

                do {
                    option = null;
                    elements = null;
                    while (! (option = $re(/^(all)(?=\s*(\)|,))/))) {
                        e = this.element();
                        if (!e) { break; }
                        if (elements) { elements.push(e); } else { elements = [ e ]; }
                    }

                    option = option && option[1];
                    if (!elements)
                        error("Missing target selector for :extend().");
                    extend = new(tree.Extend)(new(tree.Selector)(elements), option, index);
                    if (extendList) { extendList.push(extend); } else { extendList = [ extend ]; }

                } while($char(","));

                expect(/^\)/);

                if (isRule) {
                    expect(/^;/);
                }

                return extendList;
            },

            //
            // extendRule - used in a rule to extend all the parent selectors
            //
            extendRule: function() {
                return this.extend(true);
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
                    var s = input.charAt(i), important = false, index = i, elemIndex,
                        elements, elem, e, c, args;

                    if (s !== '.' && s !== '#') { return; }

                    save(); // stop us absorbing part of an invalid selector

                    while (true) {
                        elemIndex = i;
                        e = $re(/^[#.](?:[\w-]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+/);
                        if (!e) {
                            break;
                        }
                        elem = new(tree.Element)(c, e, elemIndex, env.currentFileInfo);
                        if (elements) { elements.push(elem); } else { elements = [ elem ]; }
                        c = $char('>');
                    }

                    if (elements) {
                        if ($char('(')) {
                            args = this.args(true).args;
                            expectChar(')');
                        }

                        if (parsers.important()) {
                            important = true;
                        }

                        if (parsers.end()) {
                            forget();
                            return new(tree.mixin.Call)(elements, args, index, env.currentFileInfo, important);
                        }
                    }

                    restore();
                },
                args: function (isCall) {
                    var parsers = parser.parsers, entities = parsers.entities,
                        returner = { args:null, variadic: false },
                        expressions = [], argsSemiColon = [], argsComma = [],
                        isSemiColonSeperated, expressionContainsNamed, name, nameLoop, value, arg;

                    save();

                    while (true) {
                        if (isCall) {
                            arg = parsers.detachedRuleset() || parsers.expression();
                        } else {
                            parsers.comments();
                            if (input.charAt(i) === '.' && $re(/^\.{3}/)) {
                                returner.variadic = true;
                                if ($char(";") && !isSemiColonSeperated) {
                                    isSemiColonSeperated = true;
                                }
                                (isSemiColonSeperated ? argsSemiColon : argsComma)
                                    .push({ variadic: true });
                                break;
                            }
                            arg = entities.variable() || entities.literal() || entities.keyword();
                        }

                        if (!arg) {
                            break;
                        }

                        nameLoop = null;
                        if (arg.throwAwayComments) {
                            arg.throwAwayComments();
                        }
                        value = arg;
                        var val = null;

                        if (isCall) {
                            // Variable
                            if (arg.value && arg.value.length == 1) {
                                val = arg.value[0];
                            }
                        } else {
                            val = arg;
                        }

                        if (val && val instanceof tree.Variable) {
                            if ($char(':')) {
                                if (expressions.length > 0) {
                                    if (isSemiColonSeperated) {
                                        error("Cannot mix ; and , as delimiter types");
                                    }
                                    expressionContainsNamed = true;
                                }

                                // we do not support setting a ruleset as a default variable - it doesn't make sense
                                // However if we do want to add it, there is nothing blocking it, just don't error
                                // and remove isCall dependency below
                                value = (isCall && parsers.detachedRuleset()) || parsers.expression();

                                if (!value) {
                                    if (isCall) {
                                        error("could not understand value for named argument");
                                    } else {
                                        restore();
                                        returner.args = [];
                                        return returner;
                                    }
                                }
                                nameLoop = (name = val.name);
                            } else if (!isCall && $re(/^\.{3}/)) {
                                returner.variadic = true;
                                if ($char(";") && !isSemiColonSeperated) {
                                    isSemiColonSeperated = true;
                                }
                                (isSemiColonSeperated ? argsSemiColon : argsComma)
                                    .push({ name: arg.name, variadic: true });
                                break;
                            } else if (!isCall) {
                                name = nameLoop = val.name;
                                value = null;
                            }
                        }

                        if (value) {
                            expressions.push(value);
                        }

                        argsComma.push({ name:nameLoop, value:value });

                        if ($char(',')) {
                            continue;
                        }

                        if ($char(';') || isSemiColonSeperated) {

                            if (expressionContainsNamed) {
                                error("Cannot mix ; and , as delimiter types");
                            }

                            isSemiColonSeperated = true;

                            if (expressions.length > 1) {
                                value = new(tree.Value)(expressions);
                            }
                            argsSemiColon.push({ name:name, value:value });

                            name = null;
                            expressions = [];
                            expressionContainsNamed = false;
                        }
                    }

                    forget();
                    returner.args = isSemiColonSeperated ? argsSemiColon : argsComma;
                    return returner;
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
                    var name, params = [], match, ruleset, cond, variadic = false;
                    if ((input.charAt(i) !== '.' && input.charAt(i) !== '#') ||
                        peek(/^[^{]*\}/)) {
                        return;
                    }

                    save();

                    match = $re(/^([#.](?:[\w-]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+)\s*\(/);
                    if (match) {
                        name = match[1];

                        var argInfo = this.args(false);
                        params = argInfo.args;
                        variadic = argInfo.variadic;

                        // .mixincall("@{a}");
                        // looks a bit like a mixin definition..
                        // also
                        // .mixincall(@a: {rule: set;});
                        // so we have to be nice and restore
                        if (!$char(')')) {
                            furthest = i;
                            restore();
                            return;
                        }

                        parsers.comments();

                        if ($re(/^when/)) { // Guard
                            cond = expect(parsers.conditions, 'expected condition');
                        }

                        ruleset = parsers.block();

                        if (ruleset) {
                            forget();
                            return new(tree.mixin.Definition)(name, params, ruleset, cond, variadic);
                        } else {
                            restore();
                        }
                    } else {
                        forget();
                    }
                }
            },

            //
            // Entities are the smallest recognized token,
            // and can be found inside a rule's value.
            //
            entity: function () {
                var entities = this.entities;

                return entities.literal() || entities.variable() || entities.url() ||
                       entities.call()    || entities.keyword()  || entities.javascript() ||
                       this.comment();
            },

            //
            // A Rule terminator. Note that we use `peek()` to check for '}',
            // because the `block` rule will be expecting it, but we still need to make sure
            // it's there, if ';' was ommitted.
            //
            end: function () {
                return $char(';') || peekChar('}');
            },

            //
            // IE's alpha function
            //
            //     alpha(opacity=88)
            //
            alpha: function () {
                var value;

                if (! $re(/^\(opacity=/i)) { return; }
                value = $re(/^\d+/) || this.entities.variable();
                if (value) {
                    expectChar(')');
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
                var e, c, v, index = i;

                c = this.combinator();

                e = $re(/^(?:\d+\.\d+|\d+)%/) || $re(/^(?:[.#]?|:*)(?:[\w-]|[^\x00-\x9f]|\\(?:[A-Fa-f0-9]{1,6} ?|[^A-Fa-f0-9]))+/) ||
                    $char('*') || $char('&') || this.attribute() || $re(/^\([^()@]+\)/) || $re(/^[\.#](?=@)/) ||
                    this.entities.variableCurly();

                if (! e) {
                    save();
                    if ($char('(')) {
                        if ((v = this.selector()) && $char(')')) {
                            e = new(tree.Paren)(v);
                            forget();
                        } else {
                            restore();
                        }
                    } else {
                        forget();
                    }
                }

                if (e) { return new(tree.Element)(c, e, index, env.currentFileInfo); }
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
                var c = input.charAt(i);

                if (c === '/') {
                    save();
                    var slashedCombinator = $re(/^\/[a-z]+\//i);
                    if (slashedCombinator) {
                        forget();
                        return new(tree.Combinator)(slashedCombinator);
                    }
                    restore();
                }

                if (c === '>' || c === '+' || c === '~' || c === '|' || c === '^') {
                    i++;
                    if (c === '^' && input.charAt(i) === '^') {
                        c = '^^';
                        i++;
                    }
                    while (isWhitespace(input, i)) { i++; }
                    return new(tree.Combinator)(c);
                } else if (isWhitespace(input, i - 1)) {
                    return new(tree.Combinator)(" ");
                } else {
                    return new(tree.Combinator)(null);
                }
            },
            //
            // A CSS selector (see selector below)
            // with less extensions e.g. the ability to extend and guard
            //
            lessSelector: function () {
                return this.selector(true);
            },
            //
            // A CSS Selector
            //
            //     .class > div + h1
            //     li a:hover
            //
            // Selectors are made out of one or more Elements, see above.
            //
            selector: function (isLess) {
                var index = i, $re = _$re, elements, extendList, c, e, extend, when, condition;

                while ((isLess && (extend = this.extend())) || (isLess && (when = $re(/^when/))) || (e = this.element())) {
                    if (when) {
                        condition = expect(this.conditions, 'expected condition');
                    } else if (condition) {
                        error("CSS guard can only be used at the end of selector");
                    } else if (extend) {
                        if (extendList) { extendList.push(extend); } else { extendList = [ extend ]; }
                    } else {
                        if (extendList) { error("Extend can only be used at the end of selector"); }
                        c = input.charAt(i);
                        if (elements) { elements.push(e); } else { elements = [ e ]; }
                        e = null;
                    }
                    if (c === '{' || c === '}' || c === ';' || c === ',' || c === ')') {
                        break;
                    }
                }

                if (elements) { return new(tree.Selector)(elements, extendList, condition, index, env.currentFileInfo); }
                if (extendList) { error("Extend must be used to extend a selector, it cannot be used on its own"); }
            },
            attribute: function () {
                if (! $char('[')) { return; }

                var entities = this.entities,
                    key, val, op;

                if (!(key = entities.variableCurly())) {
                    key = expect(/^(?:[_A-Za-z0-9-\*]*\|)?(?:[_A-Za-z0-9-]|\\.)+/);
                }

                op = $re(/^[|~*$^]?=/);
                if (op) {
                    val = entities.quoted() || $re(/^[0-9]+%/) || $re(/^[\w-]+/) || entities.variableCurly();
                }

                expectChar(']');

                return new(tree.Attribute)(key, op, val);
            },

            //
            // The `block` rule is used by `ruleset` and `mixin.definition`.
            // It's a wrapper around the `primary` rule, with added `{}`.
            //
            block: function () {
                var content;
                if ($char('{') && (content = this.primary()) && $char('}')) {
                    return content;
                }
            },

            blockRuleset: function() {
                var block = this.block();

                if (block) {
                    block = new tree.Ruleset(null, block);
                }
                return block;
            },

            detachedRuleset: function() {
                var blockRuleset = this.blockRuleset();
                if (blockRuleset) {
                    return new tree.DetachedRuleset(blockRuleset);
                }
            },

            //
            // div, .class, body > p {...}
            //
            ruleset: function () {
                var selectors, s, rules, debugInfo;

                save();

                if (env.dumpLineNumbers) {
                    debugInfo = getDebugInfo(i, input, env);
                }

                while (true) {
                    s = this.lessSelector();
                    if (!s) {
                        break;
                    }
                    if (selectors) { selectors.push(s); } else { selectors = [ s ]; }
                    this.comments();
                    if (s.condition && selectors.length > 1) {
                        error("Guards are only currently allowed on a single selector.");
                    }
                    if (! $char(',')) { break; }
                    if (s.condition) {
                        error("Guards are only currently allowed on a single selector.");
                    }
                    this.comments();
                }

                if (selectors && (rules = this.block())) {
                    forget();
                    var ruleset = new(tree.Ruleset)(selectors, rules, env.strictImports);
                    if (env.dumpLineNumbers) {
                        ruleset.debugInfo = debugInfo;
                    }
                    return ruleset;
                } else {
                    // Backtrack
                    furthest = i;
                    restore();
                }
            },
            rule: function (tryAnonymous) {
                var name, value, startOfRule = i, c = input.charAt(startOfRule), important, merge, isVariable;

                if (c === '.' || c === '#' || c === '&') { return; }

                save();

                name = this.variable() || this.ruleProperty();
                if (name) {
                    isVariable = typeof name === "string";

                    if (isVariable) {
                        value = this.detachedRuleset();
                    }

                    this.comments();
                    if (!value) {
                        // prefer to try to parse first if its a variable or we are compressing
                        // but always fallback on the other one
                        value = !tryAnonymous && (env.compress || isVariable) ?
                            (this.value() || this.anonymousValue()) :
                            (this.anonymousValue() || this.value());

                        important = this.important();

                        // a name returned by this.ruleProperty() is always an array of the form:
                        // [string-1, ..., string-n, ""] or [string-1, ..., string-n, "+"]
                        // where each item is a tree.Keyword or tree.Variable
                        merge = !isVariable && name.pop().value;
                    }

                    if (value && this.end()) {
                        forget();
                        return new (tree.Rule)(name, value, important, merge, startOfRule, env.currentFileInfo);
                    } else {
                        furthest = i;
                        restore();
                        if (value && !tryAnonymous) {
                            return this.rule(true);
                        }
                    }
                } else {
                    forget();
                }
            },
            anonymousValue: function () {
                var match;
                match = /^([^@+\/'"*`(;{}-]*);/.exec(current);
                if (match) {
                    i += match[0].length - 1;
                    return new(tree.Anonymous)(match[1]);
                }
            },

            //
            // An @import directive
            //
            //     @import "lib";
            //
            // Depending on our environment, importing is done differently:
            // In the browser, it's an XHR request, in Node, it would be a
            // file-system operation. The function used for importing is
            // stored in `import`, which we pass to the Import constructor.
            //
            "import": function () {
                var path, features, index = i;

                var dir = $re(/^@import?\s+/);

                if (dir) {
                    var options = (dir ? this.importOptions() : null) || {};

                    if ((path = this.entities.quoted() || this.entities.url())) {
                        features = this.mediaFeatures();

                        if (!$(';')) {
                            i = index;
                            error("missing semi-colon or unrecognised media features on import");
                        }
                        features = features && new(tree.Value)(features);
                        return new(tree.Import)(path, features, options, index, env.currentFileInfo);
                    }
                    else
                    {
                        i = index;
                        error("malformed import statement");
                    }
                }
            },

            importOptions: function() {
                var o, options = {}, optionName, value;

                // list of options, surrounded by parens
                if (! $char('(')) { return null; }
                do {
                    o = this.importOption();
                    if (o) {
                        optionName = o;
                        value = true;
                        switch(optionName) {
                            case "css":
                                optionName = "less";
                                value = false;
                            break;
                            case "once":
                                optionName = "multiple";
                                value = false;
                            break;
                        }
                        options[optionName] = value;
                        if (! $char(',')) { break; }
                    }
                } while (o);
                expectChar(')');
                return options;
            },

            importOption: function() {
                var opt = $re(/^(less|css|multiple|once|inline|reference)/);
                if (opt) {
                    return opt[1];
                }
            },

            mediaFeature: function () {
                var entities = this.entities, nodes = [], e, p;
                do {
                    e = entities.keyword() || entities.variable();
                    if (e) {
                        nodes.push(e);
                    } else if ($char('(')) {
                        p = this.property();
                        e = this.value();
                        if ($char(')')) {
                            if (p && e) {
                                nodes.push(new(tree.Paren)(new(tree.Rule)(p, e, null, null, i, env.currentFileInfo, true)));
                            } else if (e) {
                                nodes.push(new(tree.Paren)(e));
                            } else {
                                return null;
                            }
                        } else { return null; }
                    }
                } while (e);

                if (nodes.length > 0) {
                    return new(tree.Expression)(nodes);
                }
            },

            mediaFeatures: function () {
                var entities = this.entities, features = [], e;
                do {
                    e = this.mediaFeature();
                    if (e) {
                        features.push(e);
                        if (! $char(',')) { break; }
                    } else {
                        e = entities.variable();
                        if (e) {
                            features.push(e);
                            if (! $char(',')) { break; }
                        }
                    }
                } while (e);

                return features.length > 0 ? features : null;
            },

            media: function () {
                var features, rules, media, debugInfo;

                if (env.dumpLineNumbers) {
                    debugInfo = getDebugInfo(i, input, env);
                }

                if ($re(/^@media/)) {
                    features = this.mediaFeatures();

                    rules = this.block();
                    if (rules) {
                        media = new(tree.Media)(rules, features, i, env.currentFileInfo);
                        if (env.dumpLineNumbers) {
                            media.debugInfo = debugInfo;
                        }
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
                var index = i, name, value, rules, nonVendorSpecificName,
                    hasIdentifier, hasExpression, hasUnknown, hasBlock = true;

                if (input.charAt(i) !== '@') { return; }

                value = this['import']() || this.media();
                if (value) {
                    return value;
                }

                save();

                name = $re(/^@[a-z-]+/);

                if (!name) { return; }

                nonVendorSpecificName = name;
                if (name.charAt(1) == '-' && name.indexOf('-', 2) > 0) {
                    nonVendorSpecificName = "@" + name.slice(name.indexOf('-', 2) + 1);
                }

                switch(nonVendorSpecificName) {
                    /*
                    case "@font-face":
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
                    */
                    case "@charset":
                        hasIdentifier = true;
                        hasBlock = false;
                        break;
                    case "@namespace":
                        hasExpression = true;
                        hasBlock = false;
                        break;
                    case "@keyframes":
                        hasIdentifier = true;
                        break;
                    case "@host":
                    case "@page":
                    case "@document":
                    case "@supports":
                        hasUnknown = true;
                        break;
                }

                this.comments();

                if (hasIdentifier) {
                    value = this.entity();
                    if (!value) {
                        error("expected " + name + " identifier");
                    }
                } else if (hasExpression) {
                    value = this.expression();
                    if (!value) {
                        error("expected " + name + " expression");
                    }
                } else if (hasUnknown) {
                    value = ($re(/^[^{;]+/) || '').trim();
                    if (value) {
                        value = new(tree.Anonymous)(value);
                    }
                }

                this.comments();

                if (hasBlock) {
                    rules = this.blockRuleset();
                }

                if (rules || (!hasBlock && value && $char(';'))) {
                    forget();
                    return new(tree.Directive)(name, value, rules, index, env.currentFileInfo,
                        env.dumpLineNumbers ? getDebugInfo(index, input, env) : null);
                }

                restore();
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
                var e, expressions = [];

                do {
                    e = this.expression();
                    if (e) {
                        expressions.push(e);
                        if (! $char(',')) { break; }
                    }
                } while(e);

                if (expressions.length > 0) {
                    return new(tree.Value)(expressions);
                }
            },
            important: function () {
                if (input.charAt(i) === '!') {
                    return $re(/^! *important/);
                }
            },
            sub: function () {
                var a, e;

                if ($char('(')) {
                    a = this.addition();
                    if (a) {
                        e = new(tree.Expression)([a]);
                        expectChar(')');
                        e.parens = true;
                        return e;
                    }
                }
            },
            multiplication: function () {
                var m, a, op, operation, isSpaced;
                m = this.operand();
                if (m) {
                    isSpaced = isWhitespace(input, i - 1);
                    while (true) {
                        if (peek(/^\/[*\/]/)) {
                            break;
                        }

                        save();

                        op = $char('/') || $char('*');

                        if (!op) { forget(); break; }

                        a = this.operand();

                        if (!a) { restore(); break; }
                        forget();

                        m.parensInOp = true;
                        a.parensInOp = true;
                        operation = new(tree.Operation)(op, [operation || m, a], isSpaced);
                        isSpaced = isWhitespace(input, i - 1);
                    }
                    return operation || m;
                }
            },
            addition: function () {
                var m, a, op, operation, isSpaced;
                m = this.multiplication();
                if (m) {
                    isSpaced = isWhitespace(input, i - 1);
                    while (true) {
                        op = $re(/^[-+]\s+/) || (!isSpaced && ($char('+') || $char('-')));
                        if (!op) {
                            break;
                        }
                        a = this.multiplication();
                        if (!a) {
                            break;
                        }

                        m.parensInOp = true;
                        a.parensInOp = true;
                        operation = new(tree.Operation)(op, [operation || m, a], isSpaced);
                        isSpaced = isWhitespace(input, i - 1);
                    }
                    return operation || m;
                }
            },
            conditions: function () {
                var a, b, index = i, condition;

                a = this.condition();
                if (a) {
                    while (true) {
                        if (!peek(/^,\s*(not\s*)?\(/) || !$char(',')) {
                            break;
                        }
                        b = this.condition();
                        if (!b) {
                            break;
                        }
                        condition = new(tree.Condition)('or', condition || a, b, index);
                    }
                    return condition || a;
                }
            },
            condition: function () {
                var entities = this.entities, index = i, negate = false,
                    a, b, c, op;

                if ($re(/^not/)) { negate = true; }
                expectChar('(');
                a = this.addition() || entities.keyword() || entities.quoted();
                if (a) {
                    op = $re(/^(?:>=|<=|=<|[<=>])/);
                    if (op) {
                        b = this.addition() || entities.keyword() || entities.quoted();
                        if (b) {
                            c = new(tree.Condition)(op, a, b, index, negate);
                        } else {
                            error('expected expression');
                        }
                    } else {
                        c = new(tree.Condition)('=', a, new(tree.Keyword)('true'), index, negate);
                    }
                    expectChar(')');
                    return $re(/^and/) ? new(tree.Condition)('and', c, this.condition()) : c;
                }
            },

            //
            // An operand is anything that can be part of an operation,
            // such as a Color, or a Variable
            //
            operand: function () {
                var entities = this.entities,
                    p = input.charAt(i + 1), negate;

                if (input.charAt(i) === '-' && (p === '@' || p === '(')) { negate = $char('-'); }
                var o = this.sub() || entities.dimension() ||
                        entities.color() || entities.variable() ||
                        entities.call();

                if (negate) {
                    o.parensInOp = true;
                    o = new(tree.Negative)(o);
                }

                return o;
            },

            //
            // Expressions either represent mathematical operations,
            // or white-space delimited Entities.
            //
            //     1px solid black
            //     @var * 2
            //
            expression: function () {
                var entities = [], e, delim;

                do {
                    e = this.addition() || this.entity();
                    if (e) {
                        entities.push(e);
                        // operations do not allow keyword "/" dimension (e.g. small/20px) so we support that here
                        if (!peek(/^\/[\/*]/)) {
                            delim = $char('/');
                            if (delim) {
                                entities.push(new(tree.Anonymous)(delim));
                            }
                        }
                    }
                } while (e);
                if (entities.length > 0) {
                    return new(tree.Expression)(entities);
                }
            },
            property: function () {
                var name = $re(/^(\*?-?[_a-zA-Z0-9-]+)\s*:/);
                if (name) {
                    return name[1];
                }
            },
            ruleProperty: function () {
                var c = current, name = [], index = [], length = 0, s, k;

                function match(re) {
                    var a = re.exec(c);
                    if (a) {
                        index.push(i + length);
                        length += a[0].length;
                        c = c.slice(a[1].length);
                        return name.push(a[1]);
                    }
                }
                function cutOutBlockComments() {
                    //match block comments
                    var a = /^\s*\/\*(?:[^*]|\*+[^\/*])*\*+\//.exec(c);
                    if (a) {
                        length += a[0].length;
                        c = c.slice(a[0].length);
                        return true;
                    }
                    return false;
                }

                match(/^(\*?)/);
                while (match(/^((?:[\w-]+)|(?:@\{[\w-]+\}))/)); // !
                while (cutOutBlockComments());
                if ((name.length > 1) && match(/^\s*((?:\+_|\+)?)\s*:/)) {
                    // at last, we have the complete match now. move forward,
                    // convert name particles to tree objects and return:
                    skipWhitespace(length);
                    if (name[0] === '') {
                        name.shift();
                        index.shift();
                    }
                    for (k = 0; k < name.length; k++) {
                        s = name[k];
                        name[k] = (s.charAt(0) !== '@')
                            ? new(tree.Keyword)(s)
                            : new(tree.Variable)('@' + s.slice(2, -1),
                                index[k], env.currentFileInfo);
                    }
                    return name;
                }
            }
        }
    };
    return parser;
};
less.Parser.serializeVars = function(vars) {
    var s = '';

    for (var name in vars) {
        if (Object.hasOwnProperty.call(vars, name)) {
            var value = vars[name];
            s += ((name[0] === '@') ? '' : '@') + name +': '+ value +
                    ((('' + value).slice(-1) === ';') ? '' : ';');
        }
    }

    return s;
};
