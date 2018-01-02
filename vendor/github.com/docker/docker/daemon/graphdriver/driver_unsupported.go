// +build !linux,!windows,!freebsd,!solaris

package graphdriver

var (
	// Slice of drivers that should be used in an order
	priority = []string{
		"unsupported",
	}
)

// GetFSMagic returns the filesystem id given the path.
func GetFSMagic(rootpath string) (FsMagic, error) {
	return FsMagicUnsupported, nil
}
