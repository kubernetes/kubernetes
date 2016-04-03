// +build linux windows

package vfs

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/chrootarchive"
	"github.com/docker/docker/pkg/system"
	"github.com/opencontainers/runc/libcontainer/label"
)

func init() {
	graphdriver.Register("vfs", Init)
}

func Init(home string, options []string) (graphdriver.Driver, error) {
	d := &Driver{
		home: home,
	}
	return graphdriver.NaiveDiffDriver(d), nil
}

type Driver struct {
	home string
}

func (d *Driver) String() string {
	return "vfs"
}

func (d *Driver) Status() [][2]string {
	return nil
}

func (d *Driver) GetMetadata(id string) (map[string]string, error) {
	return nil, nil
}

func (d *Driver) Cleanup() error {
	return nil
}

func (d *Driver) Create(id, parent string) error {
	dir := d.dir(id)
	if err := system.MkdirAll(filepath.Dir(dir), 0700); err != nil {
		return err
	}
	if err := os.Mkdir(dir, 0755); err != nil {
		return err
	}
	opts := []string{"level:s0"}
	if _, mountLabel, err := label.InitLabels(opts); err == nil {
		label.SetFileLabel(dir, mountLabel)
	}
	if parent == "" {
		return nil
	}
	parentDir, err := d.Get(parent, "")
	if err != nil {
		return fmt.Errorf("%s: %s", parent, err)
	}
	if err := chrootarchive.CopyWithTar(parentDir, dir); err != nil {
		return err
	}
	return nil
}

func (d *Driver) dir(id string) string {
	return filepath.Join(d.home, "dir", filepath.Base(id))
}

func (d *Driver) Remove(id string) error {
	if _, err := os.Stat(d.dir(id)); err != nil {
		return err
	}
	return os.RemoveAll(d.dir(id))
}

func (d *Driver) Get(id, mountLabel string) (string, error) {
	dir := d.dir(id)
	if st, err := os.Stat(dir); err != nil {
		return "", err
	} else if !st.IsDir() {
		return "", fmt.Errorf("%s: not a directory", dir)
	}
	return dir, nil
}

func (d *Driver) Put(id string) error {
	// The vfs driver has no runtime resources (e.g. mounts)
	// to clean up, so we don't need anything here
	return nil
}

func (d *Driver) Exists(id string) bool {
	_, err := os.Stat(d.dir(id))
	return err == nil
}
