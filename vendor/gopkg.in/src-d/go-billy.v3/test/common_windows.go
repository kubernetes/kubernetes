// +build windows

package test

import "os"

var (
	customMode            os.FileMode = 0666
	expectedSymlinkTarget             = "\\dir\\file"
)
