// +build linux,go1.16

package selinux

import (
	"errors"
	"io/fs"
	"os"

	"github.com/opencontainers/selinux/pkg/pwalkdir"
)

func rchcon(fpath, label string) error {
	return pwalkdir.Walk(fpath, func(p string, _ fs.DirEntry, _ error) error {
		e := setFileLabel(p, label)
		// Walk a file tree can race with removal, so ignore ENOENT.
		if errors.Is(e, os.ErrNotExist) {
			return nil
		}
		return e
	})
}
