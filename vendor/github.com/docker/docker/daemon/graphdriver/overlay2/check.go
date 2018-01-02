// +build linux

package overlay2

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"

	"github.com/docker/docker/pkg/system"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

// hasOpaqueCopyUpBug checks whether the filesystem has a bug
// which copies up the opaque flag when copying up an opaque
// directory. When this bug exists naive diff should be used.
func hasOpaqueCopyUpBug(d string) error {
	td, err := ioutil.TempDir(d, "opaque-bug-check")
	if err != nil {
		return err
	}
	defer func() {
		if err := os.RemoveAll(td); err != nil {
			logrus.Warnf("Failed to remove check directory %v: %v", td, err)
		}
	}()

	// Make directories l1/d, l2/d, l3, work, merged
	if err := os.MkdirAll(filepath.Join(td, "l1", "d"), 0755); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(td, "l2", "d"), 0755); err != nil {
		return err
	}
	if err := os.Mkdir(filepath.Join(td, "l3"), 0755); err != nil {
		return err
	}
	if err := os.Mkdir(filepath.Join(td, "work"), 0755); err != nil {
		return err
	}
	if err := os.Mkdir(filepath.Join(td, "merged"), 0755); err != nil {
		return err
	}

	// Mark l2/d as opaque
	if err := system.Lsetxattr(filepath.Join(td, "l2", "d"), "trusted.overlay.opaque", []byte("y"), 0); err != nil {
		return errors.Wrap(err, "failed to set opaque flag on middle layer")
	}

	opts := fmt.Sprintf("lowerdir=%s:%s,upperdir=%s,workdir=%s", path.Join(td, "l2"), path.Join(td, "l1"), path.Join(td, "l3"), path.Join(td, "work"))
	if err := unix.Mount("overlay", filepath.Join(td, "merged"), "overlay", 0, opts); err != nil {
		return errors.Wrap(err, "failed to mount overlay")
	}
	defer func() {
		if err := unix.Unmount(filepath.Join(td, "merged"), 0); err != nil {
			logrus.Warnf("Failed to unmount check directory %v: %v", filepath.Join(td, "merged"), err)
		}
	}()

	// Touch file in d to force copy up of opaque directory "d" from "l2" to "l3"
	if err := ioutil.WriteFile(filepath.Join(td, "merged", "d", "f"), []byte{}, 0644); err != nil {
		return errors.Wrap(err, "failed to write to merged directory")
	}

	// Check l3/d does not have opaque flag
	xattrOpaque, err := system.Lgetxattr(filepath.Join(td, "l3", "d"), "trusted.overlay.opaque")
	if err != nil {
		return errors.Wrap(err, "failed to read opaque flag on upper layer")
	}
	if string(xattrOpaque) == "y" {
		return errors.New("opaque flag erroneously copied up, consider update to kernel 4.8 or later to fix")
	}

	return nil
}
