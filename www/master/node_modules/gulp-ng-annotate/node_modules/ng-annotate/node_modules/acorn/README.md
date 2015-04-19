# Acorn

[![Build Status](https://travis-ci.org/marijnh/acorn.svg?branch=master)](https://travis-ci.org/marijnh/acorn)

A tiny, fast JavaScript parser, written completely in JavaScript.

## Installation

The easiest way to install acorn is with [`npm`][npm].

[npm]: http://npmjs.org

```sh
npm install acorn
```

Alternately, download the source.

```sh
git clone https://github.com/marijnh/acorn.git
```

## Components

When run in a CommonJS (node.js) or AMD environment, exported values
appear in the interfaces exposed by the individual files, as usual.
When loaded in the browser (Acorn works in any JS-enabled browser more
recent than IE5) without any kind of module management, a single
global object `acorn` will be defined, and all the exported properties
will be added to that.

### acorn.js

This file contains the actual parser (and is what you get when you
`require("acorn")` in node.js).

**parse**`(input, options)` is used to parse a JavaScript program.
The `input` parameter is a string, `options` can be undefined or an
object setting some of the options listed below. The return value will
be an abstract syntax tree object as specified by the
[Mozilla Parser API][mozapi].

When  encountering   a  syntax   error,  the   parser  will   raise  a
`SyntaxError` object with a meaningful  message. The error object will
have a `pos` property that indicates the character offset at which the
error occurred,  and a `loc`  object that contains a  `{line, column}`
object referring to that same position.

[mozapi]: https://developer.mozilla.org/en-US/docs/SpiderMonkey/Parser_API

- **ecmaVersion**: Indicates the ECMAScript version to parse. Must be
  either 3, 5, or 6. This influences support for strict mode, the set
  of reserved words, and support for new syntax features. Default is 5.

- **strictSemicolons**: If `true`, prevents the parser from doing
  automatic semicolon insertion, and statements that do not end with
  a semicolon will generate an error. Defaults to `false`.

- **allowTrailingCommas**: If `false`, the parser will not allow
  trailing commas in array and object literals. Default is `true`.

- **forbidReserved**: If `true`, using a reserved word will generate
  an error. Defaults to `false`. When given the value `"everywhere"`,
  reserved words and keywords can also not be used as property names
  (as in Internet Explorer's old parser).
  
- **allowReturnOutsideFunction**: By default, a return statement at
  the top level raises an error. Set this to `true` to accept such
  code.

- **allowImportExportEverywhere**: By default, `import` and `export`
  declarations can only appear at a program's top level. Setting this
  option to `true` allows them anywhere where a statement is allowed.

- **locations**: When `true`, each node has a `loc` object attached
  with `start` and `end` subobjects, each of which contains the
  one-based line and zero-based column numbers in `{line, column}`
  form. Default is `false`.

- **onToken**: If a function is passed for this option, each found
  token will be passed in same format as `tokenize()` returns.

  If array is passed, each found token is pushed to it.

  Note that you are not allowed to call the parser from the
  callback—that will corrupt its internal state.

- **onComment**: If a function is passed for this option, whenever a
  comment is encountered the function will be called with the
  following parameters:

  - `block`: `true` if the comment is a block comment, false if it
    is a line comment.
  - `text`: The content of the comment.
  - `start`: Character offset of the start of the comment.
  - `end`: Character offset of the end of the comment.

  When the `locations` options is on, the `{line, column}` locations
  of the comment’s start and end are passed as two additional
  parameters.

  If array is passed for this option, each found comment is pushed
  to it as object in Esprima format:
  
  ```javascript
  {
    "type": "Line" | "Block",
    "value": "comment text",
    "range": ...,
    "loc": ...
  }
  ```

  Note that you are not allowed to call the parser from the
  callback—that will corrupt its internal state.

- **ranges**: Nodes have their start and end characters offsets
  recorded in `start` and `end` properties (directly on the node,
  rather than the `loc` object, which holds line/column data. To also
  add a [semi-standardized][range] "range" property holding a
  `[start, end]` array with the same numbers, set the `ranges` option
  to `true`.

- **program**: It is possible to parse multiple files into a single
  AST by passing the tree produced by parsing the first file as the
  `program` option in subsequent parses. This will add the toplevel
  forms of the parsed file to the "Program" (top) node of an existing
  parse tree.

- **sourceFile**: When the `locations` option is `true`, you can pass
  this option to add a `source` attribute in every node’s `loc`
  object. Note that the contents of this option are not examined or
  processed in any way; you are free to use whatever format you
  choose.

- **directSourceFile**: Like `sourceFile`, but a `sourceFile` property
  will be added directly to the nodes, rather than the `loc` object.

- **preserveParens**: If this option is `true`, parenthesized expressions
  are represented by (non-standard) `ParenthesizedExpression` nodes
  that have a single `expression` property containing the expression
  inside parentheses.

[range]: https://bugzilla.mozilla.org/show_bug.cgi?id=745678

**parseExpressionAt**`(input, offset, options)` will parse a single
expression in a string, and return its AST. It will not complain if
there is more of the string left after the expression.

**getLineInfo**`(input, offset)` can be used to get a `{line,
column}` object for a given program string and character offset.

**tokenize**`(input, options)` exports a primitive interface to
Acorn's tokenizer. The function takes an input string and options
similar to `parse` (though only some options are meaningful here), and
returns a function that can be called repeatedly to read a single
token, and returns a `{start, end, type, value}` object (with added
`loc` property when the `locations` option is enabled and `range`
property when the `ranges` option is enabled).

**tokTypes** holds an object mapping names to the token type objects
that end up in the `type` properties of tokens.

#### Note on using with [Escodegen][escodegen]

Escodegen supports generating comments from AST, attached in
Esprima-specific format. In order to simulate same format in
Acorn, consider following example:

```javascript
var comments = [], tokens = [];

var ast = acorn.parse('var x = 42; // answer', {
	// collect ranges for each node
	ranges: true,
	// collect comments in Esprima's format
	onComment: comments,
	// collect token ranges
	onToken: tokens
});

// attach comments using collected information
escodegen.attachComments(ast, comments, tokens);

// generate code
console.log(escodegen.generate(ast, {comment: true}));
// > 'var x = 42;    // answer'
```

[escodegen]: https://github.com/Constellation/escodegen

#### Using Acorn in an environment with a Content Security Policy

Some contexts, such as Chrome Web Apps, disallow run-time code evaluation.
Acorn uses `new Function` to generate fast functions that test whether
a word is in a given set, and will trigger a security error when used
in a context with such a
[Content Security Policy](http://www.html5rocks.com/en/tutorials/security/content-security-policy/#eval-too)
(see [#90](https://github.com/marijnh/acorn/issues/90) and
[#123](https://github.com/marijnh/acorn/issues/123)).

The `bin/without_eval` script can be used to generate a version of
`acorn.js` that has the generated code inlined, and can thus run
without evaluating anything. In versions of this library downloaded
from NPM, this script will be available as `acorn_csp.js`.

### acorn_loose.js ###

This file implements an error-tolerant parser. It exposes a single
function.

**parse_dammit**`(input, options)` takes the same arguments and
returns the same syntax tree as the `parse` function in `acorn.js`,
but never raises an error, and will do its best to parse syntactically
invalid code in as meaningful a way as it can. It'll insert identifier
nodes with name `"✖"` as placeholders in places where it can't make
sense of the input. Depends on `acorn.js`, because it uses the same
tokenizer. The loose parser does not support ECMAScript 6 syntax yet.

### util/walk.js ###

Implements an abstract syntax tree walker. Will store its interface in
`acorn.walk` when used without a module system.

**simple**`(node, visitors, base, state)` does a 'simple' walk over
a tree. `node` should be the AST node to walk, and `visitors` an
object with properties whose names correspond to node types in the
[Mozilla Parser API][mozapi]. The properties should contain functions
that will be called with the node object and, if applicable the state
at that point. The last two arguments are optional. `base` is a walker
algorithm, and `state` is a start state. The default walker will
simply visit all statements and expressions and not produce a
meaningful state. (An example of a use of state it to track scope at
each point in the tree.)

**ancestor**`(node, visitors, base, state)` does a 'simple' walk over
a tree, building up an array of ancestor nodes (including the current node)
and passing the array to callbacks in the `state` parameter.

**recursive**`(node, state, functions, base)` does a 'recursive'
walk, where the walker functions are responsible for continuing the
walk on the child nodes of their target node. `state` is the start
state, and `functions` should contain an object that maps node types
to walker functions. Such functions are called with `(node, state, c)`
arguments, and can cause the walk to continue on a sub-node by calling
the `c` argument on it with `(node, state)` arguments. The optional
`base` argument provides the fallback walker functions for node types
that aren't handled in the `functions` object. If not given, the
default walkers will be used.

**make**`(functions, base)` builds a new walker object by using the
walker functions in `functions` and filling in the missing ones by
taking defaults from `base`.

**findNodeAt**`(node, start, end, test, base, state)` tries to
locate a node in a tree at the given start and/or end offsets, which
satisfies the predicate `test`. `start` end `end` can be either `null`
(as wildcard) or a number. `test` may be a string (indicating a node
type) or a function that takes `(nodeType, node)` arguments and
returns a boolean indicating whether this node is interesting. `base`
and `state` are optional, and can be used to specify a custom walker.
Nodes are tested from inner to outer, so if two nodes match the
boundaries, the inner one will be preferred.

**findNodeAround**`(node, pos, test, base, state)` is a lot like
`findNodeAt`, but will match any node that exists 'around' (spanning)
the given position.

**findNodeAfter**`(node, pos, test, base, state)` is similar to
`findNodeAround`, but will match all nodes *after* the given position
(testing outer nodes before inner nodes).

## Command line interface

The `bin/acorn` utility can be used to parse a file from the command
line. It accepts as arguments its input file and the following
options:

- `--ecma3|--ecma5|--ecma6`: Sets the ECMAScript version to parse. Default is
  version 5.

- `--strictSemicolons`: Prevents the parser from doing automatic
  semicolon insertion. Statements that do not end in semicolons will
  generate an error.

- `--locations`: Attaches a "loc" object to each node with "start" and
  "end" subobjects, each of which contains the one-based line and
  zero-based column numbers in `{line, column}` form.

- `--compact`: No whitespace is used in the AST output.

- `--silent`: Do not output the AST, just return the exit status.

- `--help`: Print the usage information and quit.

The utility spits out the syntax tree as JSON data.
