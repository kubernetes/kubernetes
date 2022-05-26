//go:build go1.16
// +build go1.16

package toml

import (
	"io/fs"
)

// DecodeFS is just like Decode, except it will automatically read the contents
// of the file at `path` from a fs.FS instance.
func DecodeFS(fsys fs.FS, path string, v interface{}) (MetaData, error) {
	fp, err := fsys.Open(path)
	if err != nil {
		return MetaData{}, err
	}
	defer fp.Close()
	return NewDecoder(fp).Decode(v)
}
