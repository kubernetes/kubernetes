// +build windows

package system

import (
	"os"
	"time"
)

type Stat_t struct {
	name    string
	size    int64
	mode    os.FileMode
	modTime time.Time
	isDir   bool
}

func (s Stat_t) Name() string {
	return s.name
}

func (s Stat_t) Size() int64 {
	return s.size
}

func (s Stat_t) Mode() os.FileMode {
	return s.mode
}

func (s Stat_t) ModTime() time.Time {
	return s.modTime
}

func (s Stat_t) IsDir() bool {
	return s.isDir
}
