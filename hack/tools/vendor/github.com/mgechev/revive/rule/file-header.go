package rule

import (
	"fmt"
	"regexp"

	"github.com/mgechev/revive/lint"
)

// FileHeaderRule lints given else constructs.
type FileHeaderRule struct {
	header string
}

var (
	multiRegexp  = regexp.MustCompile("^/\\*")
	singleRegexp = regexp.MustCompile("^//")
)

// Apply applies the rule to given file.
func (r *FileHeaderRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.header == "" {
		checkNumberOfArguments(1, arguments, r.Name())
		var ok bool
		r.header, ok = arguments[0].(string)
		if !ok {
			panic(fmt.Sprintf("invalid argument for \"file-header\" rule: first argument should be a string, got %T", arguments[0]))
		}
	}

	failure := []lint.Failure{
		{
			Node:       file.AST,
			Confidence: 1,
			Failure:    "the file doesn't have an appropriate header",
		},
	}

	if len(file.AST.Comments) == 0 {
		return failure
	}

	g := file.AST.Comments[0]
	if g == nil {
		return failure
	}
	comment := ""
	for _, c := range g.List {
		text := c.Text
		if multiRegexp.MatchString(text) {
			text = text[2 : len(text)-2]
		} else if singleRegexp.MatchString(text) {
			text = text[2:]
		}
		comment += text
	}

	regex, err := regexp.Compile(r.header)
	if err != nil {
		panic(err.Error())
	}

	if !regex.MatchString(comment) {
		return failure
	}
	return nil
}

// Name returns the rule name.
func (r *FileHeaderRule) Name() string {
	return "file-header"
}
