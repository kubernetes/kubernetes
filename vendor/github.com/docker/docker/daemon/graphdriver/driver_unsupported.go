// +build !linux,!windows

package graphdriver

var (
	// Slice of drivers that should be used in an order
	priority = []string{
		"unsupported",
	}
)

func GetFSMagic(rootpath string) (FsMagic, error) {
	return FsMagicUnsupported, nil
}
