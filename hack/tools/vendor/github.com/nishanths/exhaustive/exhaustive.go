/*
Package exhaustive provides an analyzer that checks exhaustiveness of enum
switch statements in Go source code.

Definition of enum

The Go language spec does not provide an explicit definition for an enum. For
the purpose of this analyzer, an enum type is any named type (a.k.a. defined
type) whose underlying type is an integer (includes byte and rune), a float, or
a string type. An enum type has associated with it constants of this named type;
these constants constitute the enum members.

In the example below, Biome is an enum type with 3 members.

    type Biome int

    const (
        Tundra  Biome = 1
        Savanna Biome = 2
        Desert  Biome = 3
    )

For a constant to be an enum member for an enum type, the constant must be
declared in the same scope as the enum type. Note that the scope requirement
implies that only constants declared in the same package as the enum type's
package can constitute the enum members for the enum type.

Enum member constants for a given enum type don't necessarily have to all be
declared in the same const block. Constant values may be specified using iota,
using explicit values, or by any means of declaring a valid Go const. It is
allowed for multiple enum member constants for a given enum type to have the
same constant value.

Definition of exhaustiveness

A switch statement that switches on a value of an enum type is exhaustive if all
of the enum type's members are listed in the switch statement's cases. If
multiple enum member constants have the same constant value, it is sufficient
for any one of these same-valued members to be listed.

For an enum type defined in the same package as the switch statement, both
exported and unexported enum members must be listed to satisfy exhaustiveness.
For an enum type defined in an external package, it is sufficient that only
exported enum members are listed.

Only identifiers denoting constants (e.g. Tundra) and qualified identifiers
denoting constants (e.g. somepkg.Grassland) listed in a switch statement's cases
can contribute towards satisfying exhaustiveness. Literal values, struct fields,
re-assignable variables, etc. will not.

Type aliases

The analyzer handles type aliases for an enum type in the following manner.
Consider the example below. T2 is a enum type, and T1 is an alias for T2. Note
that we don't term T1 itself an enum type; it is only an alias for an enum
type.

    package pkg
    type T1 = newpkg.T2
    const (
        A = newpkg.A
        B = newpkg.B
    )

    package newpkg
    type T2 int
    const (
        A T2 = 1
        B T2 = 2
    )

Then a switch statement that switches on a value of type T1 (which, in reality,
is just an alternate spelling for type T2) is exhaustive if all of T2's enum
members are listed in the switch statement's cases. The same conditions
described in the previous section for same-valued enum members and for
exported/unexported enum members apply here too.

It is worth noting that, though T1 and T2 are identical types, only constants
declared in the same scope as type T2's scope can be T2's enum members. In the
example, newpkg.A and newpkg.B are T2's enum members.

The analyzer guarantees that introducing a type alias (such as type T1 =
newpkg.T2) will never result in new diagnostics from the analyzer, as long as
the set of enum member constant values of the new RHS type (newpkg.T2) is a
subset of the set of enum member constant values of the old LHS type (T1).

Advanced notes

Non-enum member constants in a switch statement's cases: Recall from an earlier
section that a constant must be declared in the same scope as the enum type to
be an enum member. It is valid, however, both to the Go type checker and to this
analyzer, for any constant of the right type to be listed in the cases of an
enum switch statement (it does not necessarily have to be an enum member
constant declared in the same scope/package as the enum type's scope/package).
This is particularly useful when a type alias is involved: A forwarding constant
declaration (such as pkg.A, in type T1's package) can take the place of the
actual enum member constant (newpkg.A, in type T2's package) in the switch
statement's cases to satisfy exhaustiveness.

    var v pkg.T1 = pkg.ReturnsT1() // v is effectively of type newpkg.T2 due to alias
    switch v {
    case pkg.A: // valid substitute for newpkg.A (same constant value)
    case pkg.B: // valid substitute for newpkg.B (same constant value)
    }

Flags

Notable flags supported by the analyzer are described below.
All of these flags are optional.

    flag                            type    default value

    -check-generated                bool    false
    -default-signifies-exhaustive   bool    false
    -ignore-enum-members            string  (none)
    -package-scope-only             bool    false

If the -check-generated flag is enabled, switch statements in generated Go
source files are also checked. Otherwise, by default, switch statements in
generated files are not checked. See https://golang.org/s/generatedcode for the
definition of generated file.

If the -default-signifies-exhaustive flag is enabled, the presence of a
'default' case in a switch statement always satisfies exhaustiveness, even if
all enum members are not listed. It is not recommended that you enable this
flag; enabling it generally defeats the purpose of exhaustiveness checking.

The -ignore-enum-members flag specifies a regular expression in Go syntax. Enum
members matching the regular expression don't have to be listed in switch
statement cases to satisfy exhaustiveness. The specified regular expression is
matched against an enum member name inclusive of the enum package import path:
for example, if the enum package import path is "example.com/pkg" and the member
name is "Tundra", the specified regular expression will be matched against the
string "example.com/pkg.Tundra".

If the -package-scope-only flag is enabled, the analyzer only finds enums
defined in package scopes, and consequently only switch statements that switch
on package-scoped enums will be checked for exhaustiveness. By default, the
analyzer finds enums defined in all scopes, and checks switch statements that
switch on all these enums.

Skip analysis

To skip checking of a specific switch statement, associate the comment shown in
the example below with the switch statement. Note the lack of whitespace between
the comment marker ("//") and the comment text ("exhaustive:ignore").

    //exhaustive:ignore
    switch v { ... }

To ignore specific enum members, see the -ignore-enum-members flag.

Switch statements in generated Go source files are not checked by default.
Use the -check-generated flag to change this behavior.
*/
package exhaustive

import (
	"flag"
	"regexp"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

var _ flag.Value = (*regexpFlag)(nil)

// regexpFlag implements the flag.Value interface for parsing
// regular expression flag values.
type regexpFlag struct{ r *regexp.Regexp }

func (v *regexpFlag) String() string {
	if v == nil || v.r == nil {
		return ""
	}
	return v.r.String()
}

func (v *regexpFlag) Set(expr string) error {
	if expr == "" {
		v.r = nil
		return nil
	}

	r, err := regexp.Compile(expr)
	if err != nil {
		return err
	}

	v.r = r
	return nil
}

func (v *regexpFlag) value() *regexp.Regexp { return v.r }

func init() {
	Analyzer.Flags.BoolVar(&fCheckGeneratedFiles, CheckGeneratedFlag, false, "check switch statements in generated files")
	Analyzer.Flags.BoolVar(&fDefaultSignifiesExhaustive, DefaultSignifiesExhaustiveFlag, false, "presence of \"default\" case in switch statements satisfies exhaustiveness, even if all enum members are not listed")
	Analyzer.Flags.Var(&fIgnoreEnumMembers, IgnoreEnumMembersFlag, "enum members matching `regex` do not have to be listed in switch statements to satisfy exhaustiveness")
	Analyzer.Flags.BoolVar(&fPackageScopeOnly, PackageScopeOnlyFlag, false, "consider enums only in package scopes, not in inner scopes")

	var unused string
	Analyzer.Flags.StringVar(&unused, IgnorePatternFlag, "", "no effect (deprecated); see -"+IgnoreEnumMembersFlag+" instead")
	Analyzer.Flags.StringVar(&unused, CheckingStrategyFlag, "", "no effect (deprecated)")
}

// Flag names used by the analyzer. They are exported for use by analyzer
// driver programs.
const (
	CheckGeneratedFlag             = "check-generated"
	DefaultSignifiesExhaustiveFlag = "default-signifies-exhaustive"
	IgnoreEnumMembersFlag          = "ignore-enum-members"
	PackageScopeOnlyFlag           = "package-scope-only"

	IgnorePatternFlag    = "ignore-pattern"    // Deprecated: see IgnoreEnumMembersFlag instead.
	CheckingStrategyFlag = "checking-strategy" // Deprecated.
)

var (
	fCheckGeneratedFiles        bool
	fDefaultSignifiesExhaustive bool
	fIgnoreEnumMembers          regexpFlag
	fPackageScopeOnly           bool
)

// resetFlags resets the flag variables to their default values.
// Useful in tests.
func resetFlags() {
	fCheckGeneratedFiles = false
	fDefaultSignifiesExhaustive = false
	fIgnoreEnumMembers = regexpFlag{}
	fPackageScopeOnly = false
}

var Analyzer = &analysis.Analyzer{
	Name:      "exhaustive",
	Doc:       "check exhaustiveness of enum switch statements",
	Run:       run,
	Requires:  []*analysis.Analyzer{inspect.Analyzer},
	FactTypes: []analysis.Fact{&enumMembersFact{}},
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	for typ, members := range findEnums(fPackageScopeOnly, pass.Pkg, inspect, pass.TypesInfo) {
		exportFact(pass, typ, members)
	}

	cfg := config{
		defaultSignifiesExhaustive: fDefaultSignifiesExhaustive,
		checkGeneratedFiles:        fCheckGeneratedFiles,
		ignoreEnumMembers:          fIgnoreEnumMembers.value(),
	}
	checkSwitchStatements(pass, inspect, cfg)
	return nil, nil
}
