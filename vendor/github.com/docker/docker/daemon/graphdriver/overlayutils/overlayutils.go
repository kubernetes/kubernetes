// +build linux

package overlayutils

import (
	"fmt"

	"github.com/docker/docker/daemon/graphdriver"
)

// ErrDTypeNotSupported denotes that the backing filesystem doesn't support d_type.
func ErrDTypeNotSupported(driver, backingFs string) error {
	msg := fmt.Sprintf("%s: the backing %s filesystem is formatted without d_type support, which leads to incorrect behavior.", driver, backingFs)
	if backingFs == "xfs" {
		msg += " Reformat the filesystem with ftype=1 to enable d_type support."
	}
	msg += " Backing filesystems without d_type support are not supported."

	return graphdriver.NotSupportedError(msg)
}
