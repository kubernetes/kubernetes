package system

import "os"

// Lstat calls os.Lstat to get a fileinfo interface back.
// This is then copied into our own locally defined structure.
func Lstat(path string) (*StatT, error) {
	fi, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}

	return fromStatT(&fi)
}
