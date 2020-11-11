/*
Package pattern implements a simple language for pattern matching Go ASTs.

Design decisions and trade-offs

The language is designed specifically for the task of filtering ASTs
to simplify the implementation of analyses in staticcheck.
It is also intended to be trivial to parse and execute.

To that end, we make certain decisions that make the language more
suited to its task, while making certain queries infeasible.

Furthermore, it is fully expected that the majority of analyses will still require ordinary Go code
to further process the filtered AST, to make use of type information and to enforce complex invariants.
It is not our goal to design a scripting language for writing entire checks in.

The language

At its core, patterns are a representation of Go ASTs, allowing for the use of placeholders to enable pattern matching.
Their syntax is inspired by LISP and Haskell, but unlike LISP, the core unit of patterns isn't the list, but the node.
There is a fixed set of nodes, identified by name, and with the exception of the Or node, all nodes have a fixed number of arguments.
In addition to nodes, there are atoms, which represent basic units such as strings or the nil value.

Pattern matching is implemented via bindings, represented by the Binding node.
A Binding can match nodes and associate them with names, to later recall the nodes.
This allows for expressing "this node must be equal to that node" constraints.

To simplify writing and reading patterns, a small amount of additional syntax exists on top of nodes and atoms.
This additional syntax doesn't add any new features of its own, it simply provides shortcuts to creating nodes and atoms.

To show an example of a pattern, first consider this snippet of Go code:

	if x := fn(); x != nil {
		for _, v := range x {
			println(v, x)
		}
	}

The corresponding AST expressed as an idiomatic pattern would look as follows:

	(IfStmt
		(AssignStmt (Ident "x") ":=" (CallExpr (Ident "fn") []))
		(BinaryExpr (Ident "x") "!=" (Ident "nil"))
		(RangeStmt
			(Ident "_") (Ident "v") ":=" (Ident "x")
			(CallExpr (Ident "println") [(Ident "v") (Ident "x")]))
		nil)

Two things are worth noting about this representation.
First, the [el1 el2 ...] syntax is a short-hand for creating lists.
It is a short-hand for el1:el2:[], which itself is a short-hand for (List el1 (List el2 (List nil nil)).
Second, note the absence of a lot of lists in places that normally accept lists.
For example, assignment assigns a number of right-hands to a number of left-hands, yet our AssignStmt is lacking any form of list.
This is due to the fact that a single node can match a list of exactly one element.
Thus, the two following forms have identical matching behavior:

	(AssignStmt (Ident "x") ":=" (CallExpr (Ident "fn") []))
	(AssignStmt [(Ident "x")] ":=" [(CallExpr (Ident "fn") [])])

This section serves as an overview of the language's syntax.
More in-depth explanations of the matching behavior as well as an exhaustive list of node types follows in the coming sections.

Pattern matching

TODO write about pattern matching

- inspired by haskell syntax, but much, much simpler and naive

Node types

The language contains two kinds of nodes: those that map to nodes in the AST, and those that implement additional logic.

Nodes that map directly to AST nodes are named identically to the types in the go/ast package.
What follows is an exhaustive list of these nodes:

	(ArrayType len elt)
	(AssignStmt lhs tok rhs)
	(BasicLit kind value)
	(BinaryExpr x op y)
	(BranchStmt tok label)
	(CallExpr fun args)
	(CaseClause list body)
	(ChanType dir value)
	(CommClause comm body)
	(CompositeLit type elts)
	(DeferStmt call)
	(Ellipsis elt)
	(EmptyStmt)
	(Field names type tag)
	(ForStmt init cond post body)
	(FuncDecl recv name type body)
	(FuncLit type body)
	(FuncType params results)
	(GenDecl specs)
	(GoStmt call)
	(Ident name)
	(IfStmt init cond body else)
	(ImportSpec name path)
	(IncDecStmt x tok)
	(IndexExpr x index)
	(InterfaceType methods)
	(KeyValueExpr key value)
	(MapType key value)
	(RangeStmt key value tok x body)
	(ReturnStmt results)
	(SelectStmt body)
	(SelectorExpr x sel)
	(SendStmt chan value)
	(SliceExpr x low high max)
	(StarExpr x)
	(StructType fields)
	(SwitchStmt init tag body)
	(TypeAssertExpr)
	(TypeSpec name type)
	(TypeSwitchStmt init assign body)
	(UnaryExpr op x)
	(ValueSpec names type values)

Additionally, there are the String, Token and nil atoms.
Strings are double-quoted string literals, as in (Ident "someName").
Tokens are also represented as double-quoted string literals, but are converted to token.Token values in contexts that require tokens,
such as in (BinaryExpr x "<" y), where "<" is transparently converted to token.LSS during matching.
The keyword 'nil' denotes the nil value, which represents the absence of any value.

We also defines the (List head tail) node, which is used to represent sequences of elements as a singly linked list.
The head is a single element, and the tail is the remainder of the list.
For example,

	(List "foo" (List "bar" (List "baz" (List nil nil))))

represents a list of three elements, "foo", "bar" and "baz". There is dedicated syntax for writing lists, which looks as follows:

	["foo" "bar" "baz"]

This syntax is itself syntactic sugar for the following form:

	"foo":"bar":"baz":[]

This form is of particular interest for pattern matching, as it allows matching on the head and tail. For example,

	"foo":"bar":_

would match any list with at least two elements, where the first two elements are "foo" and "bar". This is equivalent to writing

	(List "foo" (List "bar" _))

Note that it is not possible to match from the end of the list.
That is, there is no way to express a query such as "a list of any length where the last element is foo".

Note that unlike in LISP, nil and empty lists are distinct from one another.
In patterns, with respect to lists, nil is akin to Go's untyped nil.
It will match a nil ast.Node, but it will not match a nil []ast.Expr. Nil will, however, match pointers to named types such as *ast.Ident.
Similarly, lists are akin to Go's
slices. An empty list will match both a nil and an empty []ast.Expr, but it will not match a nil ast.Node.

Due to the difference between nil and empty lists, an empty list is represented as (List nil nil), i.e. a list with no head or tail.
Similarly, a list of one element is represented as (List el (List nil nil)). Unlike in LISP, it cannot be represented by (List el nil).

Finally, there are nodes that implement special logic or matching behavior.

(Any) matches any value. The underscore (_) maps to this node, making the following two forms equivalent:

	(Ident _)
	(Ident (Any))

(Builtin name) matches a built-in identifier or function by name.
This is a type-aware variant of (Ident name).
Instead of only comparing the name, it resolves the object behind the name and makes sure it's a pre-declared identifier.

For example, in the following piece of code

	func fn() {
		println(true)
		true := false
		println(true)
	}

the pattern

	(Builtin "true")

will match exactly once, on the first use of 'true' in the function.
Subsequent occurrences of 'true' no longer refer to the pre-declared identifier.

(Object name) matches an identifier by name, but yields the
types.Object it refers to.

(Function name) matches ast.Idents and ast.SelectorExprs that refer to a function with a given fully qualified name.
For example, "net/url.PathEscape" matches the PathEscape function in the net/url package,
and "(net/url.EscapeError).Error" refers to the Error method on the net/url.EscapeError type,
either on an instance of the type, or on the type itself.

For example, the following patterns match the following lines of code:

	(CallExpr (Function "fmt.Println") _) // pattern 1
	(CallExpr (Function "(net/url.EscapeError).Error") _) // pattern 2

	fmt.Println("hello, world") // matches pattern 1
	var x url.EscapeError
	x.Error() // matches pattern 2
	(url.EscapeError).Error(x) // also matches pattern 2

(Binding name node) creates or uses a binding.
Bindings work like variable assignments, allowing referring to already matched nodes.
As an example, bindings are necessary to match self-assignment of the form "x = x",
since we need to express that the right-hand side is identical to the left-hand side.

If a binding's node is not nil, the matcher will attempt to match a node according to the pattern.
If a binding's node is nil, the binding will either recall an existing value, or match the Any node.
It is an error to provide a non-nil node to a binding that has already been bound.

Referring back to the earlier example, the following pattern will match self-assignment of idents:

	(AssignStmt (Binding "lhs" (Ident _)) "=" (Binding "lhs" nil))

Because bindings are a crucial component of pattern matching, there is special syntax for creating and recalling bindings.
Lower-case names refer to bindings. If standing on its own, the name "foo" will be equivalent to (Binding "foo" nil).
If a name is followed by an at-sign (@) then it will create a binding for the node that follows.
Together, this allows us to rewrite the earlier example as follows:

	(AssignStmt lhs@(Ident _) "=" lhs)

(Or nodes...) is a variadic node that tries matching each node until one succeeds. For example, the following pattern matches all idents of name "foo" or "bar":

	(Ident (Or "foo" "bar"))

We could also have written

	(Or (Ident "foo") (Ident "bar"))

and achieved the same result. We can also mix different kinds of nodes:

	(Or (Ident "foo") (CallExpr (Ident "bar") _))

When using bindings inside of nodes used inside Or, all or none of the bindings will be bound.
That is, partially matched nodes that ultimately failed to match will not produce any bindings observable outside of the matching attempt.
We can thus write

	(Or (Ident name) (CallExpr name))

and 'name' will either be a String if the first option matched, or an Ident or SelectorExpr if the second option matched.

(Not node)

The Not node negates a match. For example, (Not (Ident _)) will match all nodes that aren't identifiers.

ChanDir(0)

Automatic unnesting of AST nodes

The Go AST has several types of nodes that wrap other nodes.
To simplify matching, we automatically unwrap some of these nodes.

These nodes are ExprStmt (for using expressions in a statement context),
ParenExpr (for parenthesized expressions),
DeclStmt (for declarations in a statement context),
and LabeledStmt (for labeled statements).

Thus, the query

	(FuncLit _ [(CallExpr _ _)]

will match a function literal containing a single function call,
even though in the actual Go AST, the CallExpr is nested inside an ExprStmt,
as function bodies are made up of sequences of statements.

On the flip-side, there is no way to specifically match these wrapper nodes.
For example, there is no way of searching for unnecessary parentheses, like in the following piece of Go code:

	((x)) += 2

*/
package pattern
