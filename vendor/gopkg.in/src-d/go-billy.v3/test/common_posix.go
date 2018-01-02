// +build !windows

package test

import "os"

var (
	customMode            os.FileMode = 0755
	expectedSymlinkTarget             = "/dir/file"
)
