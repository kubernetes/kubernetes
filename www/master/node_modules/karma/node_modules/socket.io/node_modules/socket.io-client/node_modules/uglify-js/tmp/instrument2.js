// sample on how to use the parser and walker API to instrument some code

var jsp = require("uglify-js").parser;
var pro = require("uglify-js").uglify;

function instrument(code) {
        var ast = jsp.parse(code, false, true); // true for the third arg specifies that we want
                                                // to have start/end tokens embedded in the
                                                // statements
        var w = pro.ast_walker();

        function trace (line, comment) {
                var code = pro.gen_code(line, { beautify: true });
                var data = line[0]

                var args = []
                if (!comment) comment = ""
                if (typeof data === "object") {
                        code = code.split(/\n/).shift()
                        args = [ [ "string", data.toString() ],
                                 [ "string", code ],
                                 [ "num", data.start.line ],
                                 [ "num", data.start.col ],
                                 [ "num", data.end.line ],
                                 [ "num", data.end.col ]]
                } else {
                        args = [ [ "string", data ],
                                 [ "string", code ]]

                }
                return [ "call", [ "name", "trace" ], args ];
        }

        // we're gonna need this to push elements that we're currently looking at, to avoid
        // endless recursion.
        var analyzing = [];
        function do_stat() {
                var ret;
                if (this[0].start && analyzing.indexOf(this) < 0) {
                        // without the `analyzing' hack, w.walk(this) would re-enter here leading
                        // to infinite recursion
                        analyzing.push(this);
                        ret = [ "splice",
                                [ [ "stat", trace(this) ],
                                  w.walk(this) ]];
                        analyzing.pop(this);
                }
                return ret;
        }

        function do_cond(c, t, f) {
                return [ this[0], w.walk(c),
                         ["seq", trace(t), w.walk(t) ],
                         ["seq", trace(f), w.walk(f) ]];
        }

        function do_binary(c, l, r) {
                if (c !== "&&" && c !== "||") {
                        return [this[0], c, w.walk(l), w.walk(r)];
                }
                return [ this[0], c,
                         ["seq", trace(l), w.walk(l) ],
                         ["seq", trace(r), w.walk(r) ]];
        }

        var new_ast = w.with_walkers({
                "stat"        : do_stat,
                "label"       : do_stat,
                "break"       : do_stat,
                "continue"    : do_stat,
                "debugger"    : do_stat,
                "var"         : do_stat,
                "const"       : do_stat,
                "return"      : do_stat,
                "throw"       : do_stat,
                "try"         : do_stat,
                "defun"       : do_stat,
                "if"          : do_stat,
                "while"       : do_stat,
                "do"          : do_stat,
                "for"         : do_stat,
                "for-in"      : do_stat,
                "switch"      : do_stat,
                "with"        : do_stat,
                "conditional" : do_cond,
                "binary"      : do_binary
        }, function(){
                return w.walk(ast);
        });
        return pro.gen_code(new_ast, { beautify: true });
}


////// test code follows.

var code = instrument(test.toString());
console.log(code);

function test() {
        // simple stats
        a = 5;
        c += a + b;
        "foo";

        // var
        var foo = 5;
        const bar = 6, baz = 7;

        // switch block.  note we can't track case lines the same way.
        switch ("foo") {
            case "foo":
                return 1;
            case "bar":
                return 2;
        }

        // for/for in
        for (var i = 0; i < 5; ++i) {
                console.log("Hello " + i);
        }
        for (var i in [ 1, 2, 3]) {
                console.log(i);
        }

        for (var i = 0; i < 5; ++i)
                console.log("foo");

        for (var i = 0; i < 5; ++i) {
                console.log("foo");
        }

        var k = plurp() ? 1 : 0;
        var x = a ? doX(y) && goZoo("zoo")
              : b ? blerg({ x: y })
              : null;

        var x = X || Y;
}
