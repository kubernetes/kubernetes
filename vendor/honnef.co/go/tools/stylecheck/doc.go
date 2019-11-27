package stylecheck

import "honnef.co/go/tools/lint"

var Docs = map[string]*lint.Documentation{
	"ST1000": &lint.Documentation{
		Title: `Incorrect or missing package comment`,
		Text: `Packages must have a package comment that is formatted according to
the guidelines laid out in
https://github.com/golang/go/wiki/CodeReviewComments#package-comments.`,
		Since:      "2019.1",
		NonDefault: true,
	},

	"ST1001": &lint.Documentation{
		Title: `Dot imports are discouraged`,
		Text: `Dot imports that aren't in external test packages are discouraged.

The dot_import_whitelist option can be used to whitelist certain
imports.

Quoting Go Code Review Comments:

    The import . form can be useful in tests that, due to circular
    dependencies, cannot be made part of the package being tested:

        package foo_test

        import (
            "bar/testutil" // also imports "foo"
            . "foo"
        )

    In this case, the test file cannot be in package foo because it
    uses bar/testutil, which imports foo. So we use the 'import .'
    form to let the file pretend to be part of package foo even though
    it is not. Except for this one case, do not use import . in your
    programs. It makes the programs much harder to read because it is
    unclear whether a name like Quux is a top-level identifier in the
    current package or in an imported package.`,
		Since:   "2019.1",
		Options: []string{"dot_import_whitelist"},
	},

	"ST1003": &lint.Documentation{
		Title: `Poorly chosen identifier`,
		Text: `Identifiers, such as variable and package names, follow certain rules.

See the following links for details:

- https://golang.org/doc/effective_go.html#package-names
- https://golang.org/doc/effective_go.html#mixed-caps
- https://github.com/golang/go/wiki/CodeReviewComments#initialisms
- https://github.com/golang/go/wiki/CodeReviewComments#variable-names`,
		Since:      "2019.1",
		NonDefault: true,
		Options:    []string{"initialisms"},
	},

	"ST1005": &lint.Documentation{
		Title: `Incorrectly formatted error string`,
		Text: `Error strings follow a set of guidelines to ensure uniformity and good
composability.

Quoting Go Code Review Comments:

    Error strings should not be capitalized (unless beginning with
    proper nouns or acronyms) or end with punctuation, since they are
    usually printed following other context. That is, use
    fmt.Errorf("something bad") not fmt.Errorf("Something bad"), so
    that log.Printf("Reading %s: %v", filename, err) formats without a
    spurious capital letter mid-message.`,
		Since: "2019.1",
	},

	"ST1006": &lint.Documentation{
		Title: `Poorly chosen receiver name`,
		Text: `Quoting Go Code Review Comments:

    The name of a method's receiver should be a reflection of its
    identity; often a one or two letter abbreviation of its type
    suffices (such as "c" or "cl" for "Client"). Don't use generic
    names such as "me", "this" or "self", identifiers typical of
    object-oriented languages that place more emphasis on methods as
    opposed to functions. The name need not be as descriptive as that
    of a method argument, as its role is obvious and serves no
    documentary purpose. It can be very short as it will appear on
    almost every line of every method of the type; familiarity admits
    brevity. Be consistent, too: if you call the receiver "c" in one
    method, don't call it "cl" in another.`,
		Since: "2019.1",
	},

	"ST1008": &lint.Documentation{
		Title: `A function's error value should be its last return value`,
		Text:  `A function's error value should be its last return value.`,
		Since: `2019.1`,
	},

	"ST1011": &lint.Documentation{
		Title: `Poorly chosen name for variable of type time.Duration`,
		Text: `time.Duration values represent an amount of time, which is represented
as a count of nanoseconds. An expression like 5 * time.Microsecond
yields the value 5000. It is therefore not appropriate to suffix a
variable of type time.Duration with any time unit, such as Msec or
Milli.`,
		Since: `2019.1`,
	},

	"ST1012": &lint.Documentation{
		Title: `Poorly chosen name for error variable`,
		Text: `Error variables that are part of an API should be called errFoo or
ErrFoo.`,
		Since: "2019.1",
	},

	"ST1013": &lint.Documentation{
		Title: `Should use constants for HTTP error codes, not magic numbers`,
		Text: `HTTP has a tremendous number of status codes. While some of those are
well known (200, 400, 404, 500), most of them are not. The net/http
package provides constants for all status codes that are part of the
various specifications. It is recommended to use these constants
instead of hard-coding magic numbers, to vastly improve the
readability of your code.`,
		Since:   "2019.1",
		Options: []string{"http_status_code_whitelist"},
	},

	"ST1015": &lint.Documentation{
		Title: `A switch's default case should be the first or last case`,
		Since: "2019.1",
	},

	"ST1016": &lint.Documentation{
		Title:      `Use consistent method receiver names`,
		Since:      "2019.1",
		NonDefault: true,
	},

	"ST1017": &lint.Documentation{
		Title: `Don't use Yoda conditions`,
		Text: `Yoda conditions are conditions of the kind 'if 42 == x', where the
literal is on the left side of the comparison. These are a common
idiom in languages in which assignment is an expression, to avoid bugs
of the kind 'if (x = 42)'. In Go, which doesn't allow for this kind of
bug, we prefer the more idiomatic 'if x == 42'.`,
		Since: "2019.2",
	},

	"ST1018": &lint.Documentation{
		Title: `Avoid zero-width and control characters in string literals`,
		Since: "2019.2",
	},
}
