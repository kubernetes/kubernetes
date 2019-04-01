// +build !linux,!freebsd freebsd,!cgo

package mount // import "github.com/docker/docker/pkg/mount"

func mount(device, target, mType string, flag uintptr, data string) error {
	panic("Not implemented")
}

func unmount(target string, flag int) error {
	panic("Not implemented")
}
