package packages

import (
	"fmt"
	"path/filepath"
	"regexp"
)

func pathElemReImpl(e string, sep rune) string {
	escapedSep := regexp.QuoteMeta(string(sep)) // needed for windows sep '\\'
	return fmt.Sprintf(`(^|%s)%s($|%s)`, escapedSep, e, escapedSep)
}

func pathElemRe(e string) string {
	return pathElemReImpl(e, filepath.Separator)
}

var StdExcludeDirRegexps = []string{
	pathElemRe("vendor"),
	pathElemRe("third_party"),
	pathElemRe("testdata"),
	pathElemRe("examples"),
	pathElemRe("Godeps"),
	pathElemRe("builtin"),
}
