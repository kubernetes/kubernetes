package processors

import (
	"regexp"

	"github.com/golangci/golangci-lint/pkg/result"
)

type replacePattern struct {
	re   string
	repl string
}

type replaceRegexp struct {
	re   *regexp.Regexp
	repl string
}

var replacePatterns = []replacePattern{
	// unparam
	{`^(\S+) - (\S+) is unused$`, "`${1}` - `${2}` is unused"},
	{`^(\S+) - (\S+) always receives (\S+) \((.*)\)$`, "`${1}` - `${2}` always receives `${3}` (`${4}`)"},
	{`^(\S+) - (\S+) always receives (.*)$`, "`${1}` - `${2}` always receives `${3}`"},
	{`^(\S+) - result (\S+) is always (\S+)`, "`${1}` - result `${2}` is always `${3}`"},

	// interfacer
	{`^(\S+) can be (\S+)$`, "`${1}` can be `${2}`"},

	// govet
	{`^printf: (\S+) arg list ends with redundant newline$`, "printf: `${1}` arg list ends with redundant newline"},
	{`^composites: (\S+) composite literal uses unkeyed fields$`, "composites: `${1}` composite literal uses unkeyed fields"},

	// gosec
	{`^(\S+): Blacklisted import (\S+): weak cryptographic primitive$`,
		"${1}: Blacklisted import `${2}`: weak cryptographic primitive"},
	{`^TLS InsecureSkipVerify set true.$`, "TLS `InsecureSkipVerify` set true."},

	// gosimple
	{`should replace loop with (.*)$`, "should replace loop with `${1}`"},
	{`should use a simple channel send/receive instead of select with a single case`,
		"should use a simple channel send/receive instead of `select` with a single case"},
	{`should omit comparison to bool constant, can be simplified to (.+)$`,
		"should omit comparison to bool constant, can be simplified to `${1}`"},
	{`should write (.+) instead of (.+)$`, "should write `${1}` instead of `${2}`"},
	{`redundant return statement$`, "redundant `return` statement"},
	{`should replace this if statement with an unconditional strings.TrimPrefix`,
		"should replace this `if` statement with an unconditional `strings.TrimPrefix`"},

	// staticcheck
	{`this value of (\S+) is never used$`, "this value of `${1}` is never used"},
	{`should use time.Since instead of time.Now\(\).Sub$`,
		"should use `time.Since` instead of `time.Now().Sub`"},
	{`should check returned error before deferring response.Close\(\)$`,
		"should check returned error before deferring `response.Close()`"},
	{`no value of type uint is less than 0$`, "no value of type `uint` is less than `0`"},

	// unused
	{`(func|const|field|type|var) (\S+) is unused$`, "${1} `${2}` is unused"},

	// typecheck
	{`^unknown field (\S+) in struct literal$`, "unknown field `${1}` in struct literal"},
	{`^invalid operation: (\S+) \(variable of type (\S+)\) has no field or method (\S+)$`,
		"invalid operation: `${1}` (variable of type `${2}`) has no field or method `${3}`"},
	{`^undeclared name: (\S+)$`, "undeclared name: `${1}`"},
	{`^cannot use addr \(variable of type (\S+)\) as (\S+) value in argument to (\S+)$`,
		"cannot use addr (variable of type `${1}`) as `${2}` value in argument to `${3}`"},
	{`^other declaration of (\S+)$`, "other declaration of `${1}`"},
	{`^(\S+) redeclared in this block$`, "`${1}` redeclared in this block"},

	// golint
	{`^exported (type|method|function|var|const) (\S+) should have comment or be unexported$`,
		"exported ${1} `${2}` should have comment or be unexported"},
	{`^comment on exported (type|method|function|var|const) (\S+) should be of the form "(\S+) ..."$`,
		"comment on exported ${1} `${2}` should be of the form `${3} ...`"},
	{`^should replace (.+) with (.+)$`, "should replace `${1}` with `${2}`"},
	{`^if block ends with a return statement, so drop this else and outdent its block$`,
		"`if` block ends with a `return` statement, so drop this `else` and outdent its block"},
	{`^(struct field|var|range var|const|type|(?:func|method|interface method) (?:parameter|result)) (\S+) should be (\S+)$`,
		"${1} `${2}` should be `${3}`"},
	{`^don't use underscores in Go names; var (\S+) should be (\S+)$`,
		"don't use underscores in Go names; var `${1}` should be `${2}`"},
}

type IdentifierMarker struct {
	replaceRegexps []replaceRegexp
}

func NewIdentifierMarker() *IdentifierMarker {
	var replaceRegexps []replaceRegexp
	for _, p := range replacePatterns {
		r := replaceRegexp{
			re:   regexp.MustCompile(p.re),
			repl: p.repl,
		}
		replaceRegexps = append(replaceRegexps, r)
	}

	return &IdentifierMarker{
		replaceRegexps: replaceRegexps,
	}
}

func (im IdentifierMarker) Process(issues []result.Issue) ([]result.Issue, error) {
	return transformIssues(issues, func(i *result.Issue) *result.Issue {
		iCopy := *i
		iCopy.Text = im.markIdentifiers(iCopy.Text)
		return &iCopy
	}), nil
}

func (im IdentifierMarker) markIdentifiers(s string) string {
	for _, rr := range im.replaceRegexps {
		rs := rr.re.ReplaceAllString(s, rr.repl)
		if rs != s {
			return rs
		}
	}

	return s
}

func (im IdentifierMarker) Name() string {
	return "identifier_marker"
}
func (im IdentifierMarker) Finish() {}
