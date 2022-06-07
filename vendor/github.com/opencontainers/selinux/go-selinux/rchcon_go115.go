// +build linux,!go1.16

package selinux

import (
	"errors"
	"os"

	"github.com/opencontainers/selinux/pkg/pwalk"
)

func rchcon(fpath, label string) error {
	return pwalk.Walk(fpath, func(p string, _ os.FileInfo, _ error) error {
		e := setFileLabel(p, label)
		// Walk a file tree can race with removal, so ignore ENOENT.
		if errors.Is(e, os.ErrNotExist) {
			return nil
		}
		return e
	})
}
