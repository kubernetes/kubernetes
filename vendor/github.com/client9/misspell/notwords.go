package misspell

import (
	"bytes"
	"regexp"
	"strings"
)

var (
	reEmail     = regexp.MustCompile(`[a-zA-Z0-9_.%+-]+@[a-zA-Z0-9-.]+\.[a-zA-Z]{2,6}[^a-zA-Z]`)
	reHost      = regexp.MustCompile(`[a-zA-Z0-9-.]+\.[a-zA-Z]+`)
	reBackslash = regexp.MustCompile(`\\[a-z]`)
)

// RemovePath attempts to strip away embedded file system paths, e.g.
//  /foo/bar or /static/myimg.png
//
//  TODO: windows style
//
func RemovePath(s string) string {
	out := bytes.Buffer{}
	var idx int
	for len(s) > 0 {
		if idx = strings.IndexByte(s, '/'); idx == -1 {
			out.WriteString(s)
			break
		}

		if idx > 0 {
			idx--
		}

		var chclass string
		switch s[idx] {
		case '/', ' ', '\n', '\t', '\r':
			chclass = " \n\r\t"
		case '[':
			chclass = "]\n"
		case '(':
			chclass = ")\n"
		default:
			out.WriteString(s[:idx+2])
			s = s[idx+2:]
			continue
		}

		endx := strings.IndexAny(s[idx+1:], chclass)
		if endx != -1 {
			out.WriteString(s[:idx+1])
			out.Write(bytes.Repeat([]byte{' '}, endx))
			s = s[idx+endx+1:]
		} else {
			out.WriteString(s)
			break
		}
	}
	return out.String()
}

// replaceWithBlanks returns a string with the same number of spaces as the input
func replaceWithBlanks(s string) string {
	return strings.Repeat(" ", len(s))
}

// RemoveEmail remove email-like strings, e.g. "nickg+junk@xfoobar.com", "nickg@xyz.abc123.biz"
func RemoveEmail(s string) string {
	return reEmail.ReplaceAllStringFunc(s, replaceWithBlanks)
}

// RemoveHost removes host-like strings "foobar.com" "abc123.fo1231.biz"
func RemoveHost(s string) string {
	return reHost.ReplaceAllStringFunc(s, replaceWithBlanks)
}

// RemoveBackslashEscapes removes characters that are preceeded by a backslash
// commonly found in printf format stringd "\nto"
func removeBackslashEscapes(s string) string {
	return reBackslash.ReplaceAllStringFunc(s, replaceWithBlanks)
}

// RemoveNotWords blanks out all the not words
func RemoveNotWords(s string) string {
	// do most selective/specific first
	return removeBackslashEscapes(RemoveHost(RemoveEmail(RemovePath(StripURL(s)))))
}
