# Package `regex/syntax`

Package `syntax` provides regular expressions parser as well as AST definitions.

## Rationale

There are several problems with the stdlib [regexp/syntax](https://golang.org/pkg/regexp/syntax/) package:

1. It does several transformations during the parsing that make it
   hard to do any kind of syntax analysis afterward.

2. The AST used there is optimized for the compilation and
   execution inside the [regexp](https://golang.org/pkg/regexp) package.
   It's somewhat complicated, especially in a way character ranges are encoded.

3. It only supports [re2](https://github.com/google/re2/wiki/Syntax) syntax.
   This parser recognizes most PCRE operations.

4. It's easier to extend this package than something from the standard library.

This package does almost no assumptions about how generated AST is going to be used
so it preserves as much syntax information as possible.

It's easy to write another intermediate representation on top of it. The main
function of this package is to convert a textual regexp pattern into a more
structured form that can be processed more easily.
