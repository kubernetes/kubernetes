package forbidigo

import (
	"fmt"
	"regexp"
	"regexp/syntax"
	"strings"
)

type pattern struct {
	pattern *regexp.Regexp
	msg     string
}

func parse(ptrn string) (*pattern, error) {
	ptrnRe, err := regexp.Compile(ptrn)
	if err != nil {
		return nil, fmt.Errorf("unable to compile pattern `%s`: %s", ptrn, err)
	}
	re, err := syntax.Parse(ptrn, syntax.Perl)
	if err != nil {
		return nil, fmt.Errorf("unable to parse pattern `%s`: %s", ptrn, err)
	}
	msg := extractComment(re)
	return &pattern{pattern: ptrnRe, msg: msg}, nil
}

// Traverse the leaf submatches in the regex tree and extract a comment, if any
// is present.
func extractComment(re *syntax.Regexp) string {
	for _, sub := range re.Sub {
		if len(sub.Sub) > 0 {
			if comment := extractComment(sub); comment != "" {
				return comment
			}
		}
		subStr := sub.String()
		if strings.HasPrefix(subStr, "#") {
			return strings.TrimSpace(strings.TrimPrefix(subStr, "#"))
		}
	}
	return ""
}
