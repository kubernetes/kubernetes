package rule

import (
	"bufio"
	"bytes"
	"fmt"
	"go/token"
	"strings"
	"unicode/utf8"

	"github.com/mgechev/revive/lint"
)

// LineLengthLimitRule lints given else constructs.
type LineLengthLimitRule struct {
	max int
}

// Apply applies the rule to given file.
func (r *LineLengthLimitRule) Apply(file *lint.File, arguments lint.Arguments) []lint.Failure {
	if r.max == 0 {
		checkNumberOfArguments(1, arguments, r.Name())

		max, ok := arguments[0].(int64) // Alt. non panicking version
		if !ok || max < 0 {
			panic(`invalid value passed as argument number to the "line-length-limit" rule`)
		}

		r.max = int(max)
	}

	var failures []lint.Failure
	checker := lintLineLengthNum{
		max:  r.max,
		file: file,
		onFailure: func(failure lint.Failure) {
			failures = append(failures, failure)
		},
	}

	checker.check()

	return failures
}

// Name returns the rule name.
func (r *LineLengthLimitRule) Name() string {
	return "line-length-limit"
}

type lintLineLengthNum struct {
	max       int
	file      *lint.File
	onFailure func(lint.Failure)
}

func (r lintLineLengthNum) check() {
	f := bytes.NewReader(r.file.Content())
	spaces := strings.Repeat(" ", 4) // tab width = 4
	l := 1
	s := bufio.NewScanner(f)
	for s.Scan() {
		t := s.Text()
		t = strings.ReplaceAll(t, "\t", spaces)
		c := utf8.RuneCountInString(t)
		if c > r.max {
			r.onFailure(lint.Failure{
				Category: "code-style",
				Position: lint.FailurePosition{
					// Offset not set; it is non-trivial, and doesn't appear to be needed.
					Start: token.Position{
						Filename: r.file.Name,
						Line:     l,
						Column:   0,
					},
					End: token.Position{
						Filename: r.file.Name,
						Line:     l,
						Column:   c,
					},
				},
				Confidence: 1,
				Failure:    fmt.Sprintf("line is %d characters, out of limit %d", c, r.max),
			})
		}
		l++
	}
}
