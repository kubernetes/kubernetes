// +build !windows

package fstest

import "github.com/containerd/continuity/sysx"

// SetXAttr sets the xatter for the file
func SetXAttr(name, key, value string) Applier {
	return applyFn(func(root string) error {
		return sysx.LSetxattr(name, key, []byte(value), 0)
	})
}
