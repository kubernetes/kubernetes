package builder

import (
	"regexp"
	"strings"
)

const acceptableRemoteMIME = `(?:application/(?:(?:x\-)?tar|octet\-stream|((?:x\-)?(?:gzip|bzip2?|xz)))|(?:text/plain))`

var mimeRe = regexp.MustCompile(acceptableRemoteMIME)

func selectAcceptableMIME(ct string) string {
	return mimeRe.FindString(ct)
}

func handleJsonArgs(args []string, attributes map[string]bool) []string {
	if len(args) == 0 {
		return []string{}
	}

	if attributes != nil && attributes["json"] {
		return args
	}

	// literal string command, not an exec array
	return []string{strings.Join(args, " ")}
}
