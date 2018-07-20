package containerfs

import "path/filepath"

// cleanScopedPath removes the C:\ syntax, and prepares to combine
// with a volume path
func cleanScopedPath(path string) string {
	if len(path) >= 2 {
		c := path[0]
		if path[1] == ':' && ('a' <= c && c <= 'z' || 'A' <= c && c <= 'Z') {
			path = path[2:]
		}
	}
	return filepath.Join(string(filepath.Separator), path)
}
