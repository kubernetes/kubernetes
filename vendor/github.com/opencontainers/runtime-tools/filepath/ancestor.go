package filepath

import (
	"fmt"
	"strings"
)

// IsAncestor returns true when pathB is an strict ancestor of pathA,
// and false where the paths are equal or pathB is outside of pathA.
// Paths that are not absolute will be made absolute with Abs.
func IsAncestor(os, pathA, pathB, cwd string) (_ bool, err error) {
	if pathA == pathB {
		return false, nil
	}

	pathA, err = Abs(os, pathA, cwd)
	if err != nil {
		return false, err
	}
	pathB, err = Abs(os, pathB, cwd)
	if err != nil {
		return false, err
	}
	sep := Separator(os)
	if !strings.HasSuffix(pathA, string(sep)) {
		pathA = fmt.Sprintf("%s%c", pathA, sep)
	}
	if pathA == pathB {
		return false, nil
	}
	return strings.HasPrefix(pathB, pathA), nil
}
