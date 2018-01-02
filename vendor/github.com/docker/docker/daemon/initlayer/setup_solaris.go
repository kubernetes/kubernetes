// +build solaris,cgo

package initlayer

// Setup populates a directory with mountpoints suitable
// for bind-mounting dockerinit into the container. The mountpoint is simply an
// empty file at /.dockerinit
//
// This extra layer is used by all containers as the top-most ro layer. It protects
// the container from unwanted side-effects on the rw layer.
func Setup(initLayer string, rootUID, rootGID int) error {
	return nil
}
