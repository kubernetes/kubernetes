package dsl

// Matcher is a main API group-level entry point.
// It's used to define and configure the group rules.
// It also represents a map of all rule-local variables.
type Matcher map[string]Var

// Import loads given package path into a rule group imports table.
//
// That table is used during the rules compilation.
//
// The table has the following effect on the rules:
//	* For type expressions, it's used to resolve the
//	  full package paths of qualified types, like `foo.Bar`.
//	  If Import(`a/b/foo`) is called, `foo.Bar` will match
//	  `a/b/foo.Bar` type during the pattern execution.
func (m Matcher) Import(pkgPath string) {}

// Match specifies a set of patterns that match a rule being defined.
// Pattern matching succeeds if at least 1 pattern matches.
//
// If none of the given patterns matched, rule execution stops.
func (m Matcher) Match(pattern string, alternatives ...string) Matcher {
	return m
}

// MatchComment is like Match, but handles only comments and uses regexp patterns.
//
// Multi-line /**/ comments are passed as a single string.
// Single-line // comments are passed line-by-line.
//
// Hint: if you want to match a plain text and don't want to do meta char escaping,
// prepend `\Q` to your pattern. `\Qf(x)` will match `f(x)` as a plain text
// and there is no need to escape the `(` and `)` chars.
//
// Named regexp capture groups can be accessed using the usual indexing notation.
//
// Given this pattern:
//
//     `(?P<first>\d+)\.(\d+).(?P<second>\d+)`
//
// And this input comment: `// 14.6.600`
//
// We'll get these submatches:
//
//     m["$$"] => `14.6.600`
//     m["first"] => `14`
//     m["second"] => `600`
//
// All usual filters can be applied:
//
//     Where(!m["first"].Text.Matches(`foo`))
//
// You can use this to reject some matches (allow-list behavior).
func (m Matcher) MatchComment(pattern string, alternatives ...string) Matcher {
	return m
}

// Where applies additional constraint to a match.
// If a given cond is not satisfied, a match is rejected and
// rule execution stops.
func (m Matcher) Where(cond bool) Matcher {
	return m
}

// Report prints a message if associated rule match is successful.
//
// A message is a string that can contain interpolated expressions.
// For every matched variable it's possible to interpolate
// their printed representation into the message text with $<name>.
// An entire match can be addressed with $$.
func (m Matcher) Report(message string) Matcher {
	return m
}

// Suggest assigns a quickfix suggestion for the matched code.
func (m Matcher) Suggest(suggestion string) Matcher {
	return m
}

// At binds the reported node to a named submatch.
// If no explicit location is given, the outermost node ($$) is used.
func (m Matcher) At(v Var) Matcher {
	return m
}

// File returns the current file context.
func (m Matcher) File() File { return File{} }

// GoVersion returns the analyzer associated target Go language version.
func (m Matcher) GoVersion() GoVersion { return GoVersion{} }

// Deadcode reports whether this match is contained inside a dead code path.
func (m Matcher) Deadcode() bool { return boolResult }

// Var is a pattern variable that describes a named submatch.
type Var struct {
	// Pure reports whether expr matched by var is side-effect-free.
	Pure bool

	// Const reports whether expr matched by var is a constant value.
	Const bool

	// ConstSlice reports whether expr matched by var is a slice literal
	// consisting of contant elements.
	//
	// We need a separate Const-like predicate here because Go doesn't
	// treat slices of const elements as constants, so including
	// them in Const would be incorrect.
	// Use `m["x"].Const || m["x"].ConstSlice` when you need
	// to have extended definition of "constant value".
	//
	// Some examples:
	//     []byte("foo") -- constant byte slice
	//     []byte{'f', 'o', 'o'} -- same constant byte slice
	//     []int{1, 2} -- constant int slice
	ConstSlice bool

	// Value is a compile-time computable value of the expression.
	Value ExprValue

	// Addressable reports whether the corresponding expression is addressable.
	// See https://golang.org/ref/spec#Address_operators.
	Addressable bool

	// Type is a type of a matched expr.
	//
	// For function call expressions, a type is a function result type,
	// but for a function expression itself it's a *types.Signature.
	//
	// Suppose we have a `a.b()` expression:
	//	`$x()` m["x"].Type is `a.b` function type
	//	`$x` m["x"].Type is `a.b()` function call result type
	Type ExprType

	// Object is an associated "go/types" Object.
	Object TypesObject

	// Text is a captured node text as in the source code.
	Text MatchedText

	// Node is a captured AST node.
	Node MatchedNode

	// Line is a source code line number that contains this match.
	// If this match is multi-line, this is the first line number.
	Line int
}

// Filter applies a custom predicate function on a submatch.
//
// The callback function should use VarFilterContext to access the
// information that is usually accessed through Var.
// For example, `VarFilterContext.Type` is mapped to `Var.Type`.
func (Var) Filter(pred func(*VarFilterContext) bool) bool { return boolResult }

// MatchedNode represents an AST node associated with a named submatch.
type MatchedNode struct{}

// Is reports whether a matched node AST type is compatible with the specified type.
// A valid argument is a ast.Node implementing type name from the "go/ast" package.
// Examples: "BasicLit", "Expr", "Stmt", "Ident", "ParenExpr".
// See https://golang.org/pkg/go/ast/.
func (MatchedNode) Is(typ string) bool { return boolResult }

// Parent returns a matched node parent.
func (MatchedNode) Parent() Node { return Node{} }

// Node represents an AST node somewhere inside a match.
// Unlike MatchedNode, it doesn't have to be associated with a named submatch.
type Node struct{}

// Is reports whether a node AST type is compatible with the specified type.
// See `MatchedNode.Is` for the full reference.
func (Node) Is(typ string) bool { return boolResult }

// ExprValue describes a compile-time computable value of a matched expr.
type ExprValue struct{}

// Int returns compile-time computable int value of the expression.
// If value can't be computed, condition will fail.
func (ExprValue) Int() int { return intResult }

// TypesObject is a types.Object mapping.
type TypesObject struct{}

// Is reports whether an associated types.Object is compatible with the specified type.
// A valid argument is a types.Object type name from the "go/types" package.
// Examples: "Func", "Var", "Const", "TypeName", "Label", "PkgName", "Builtin", "Nil"
// See https://golang.org/pkg/go/types/.
func (TypesObject) Is(typ string) bool { return boolResult }

// ExprType describes a type of a matcher expr.
type ExprType struct {
	// Size represents expression type size in bytes.
	Size int
}

// Underlying returns expression type underlying type.
// See https://golang.org/pkg/go/types/#Type Underlying() method documentation.
// Read https://golang.org/ref/spec#Types section to learn more about underlying types.
func (ExprType) Underlying() ExprType { return underlyingType }

// AssignableTo reports whether a type is assign-compatible with a given type.
// See https://golang.org/pkg/go/types/#AssignableTo.
func (ExprType) AssignableTo(typ string) bool { return boolResult }

// ConvertibleTo reports whether a type is conversible to a given type.
// See https://golang.org/pkg/go/types/#ConvertibleTo.
func (ExprType) ConvertibleTo(typ string) bool { return boolResult }

// Implements reports whether a type implements a given interface.
// See https://golang.org/pkg/go/types/#Implements.
func (ExprType) Implements(typ typeName) bool { return boolResult }

// Is reports whether a type is identical to a given type.
func (ExprType) Is(typ string) bool { return boolResult }

// MatchedText represents a source text associated with a matched node.
type MatchedText string

// Matches reports whether the text matches the given regexp pattern.
func (MatchedText) Matches(pattern string) bool { return boolResult }

// String represents an arbitrary string-typed data.
type String string

// Matches reports whether a string matches the given regexp pattern.
func (String) Matches(pattern string) bool { return boolResult }

// File represents the current Go source file.
type File struct {
	// Name is a file base name.
	Name String

	// PkgPath is a file package path.
	// Examples: "io/ioutil", "strings", "github.com/quasilyte/go-ruleguard/dsl".
	PkgPath String
}

// Imports reports whether the current file imports the given path.
func (File) Imports(path string) bool { return boolResult }

// GoVersion is an analysis target go language version.
// It can be compared to Go versions like "1.10", "1.16" using
// the associated methods.
type GoVersion struct{}

// Eq asserts that target Go version is equal to (==) specified version.
func (GoVersion) Eq(version string) bool { return boolResult }

// GreaterEqThan asserts that target Go version is greater or equal than (>=) specified version.
func (GoVersion) GreaterEqThan(version string) bool { return boolResult }

// GreaterThan asserts that target Go version is greater than (>) specified version.
func (GoVersion) GreaterThan(version string) bool { return boolResult }

// LessThan asserts that target Go version is less than (<) specified version.
func (GoVersion) LessThan(version string) bool { return boolResult }

// LessEqThan asserts that target Go version is less or equal than (<=) specified version.
func (GoVersion) LessEqThan(version string) bool { return boolResult }

// typeName is a helper type used to document function params better.
//
// A type name can be:
//	- builtin type name: `error`, `string`, etc.
//	- qualified name from a standard library: `io.Reader`, etc.
//	- fully-qualified type name, like `github.com/username/pkgname.TypeName`
//
// typeName is also affected by a local import table, which can override
// how qualified names are interpreted.
// See `Matcher.Import` for more info.
type typeName = string
