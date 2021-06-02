package gitignore

import (
	"os"
	"strings"
)

func cutN(path string, n int) (string, bool) {
	isLast := true

	var i, count int
	for i < len(path)-1 {
		if os.IsPathSeparator(path[i]) {
			count++
			if count >= n {
				isLast = false
				break
			}
		}
		i++
	}
	return path[:i+1], isLast
}

func cutLastN(path string, n int) (string, bool) {
	isLast := true
	i := len(path) - 1

	var count int
	for i >= 0 {
		if os.IsPathSeparator(path[i]) {
			count++
			if count >= n {
				isLast = false
				break
			}
		}
		i--
	}
	return path[i+1:], isLast
}

func hasMeta(path string) bool {
	return strings.IndexAny(path, "*?[") >= 0
}
