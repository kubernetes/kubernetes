// +build linux freebsd

package graphtest

import (
	"os"
	"syscall"
	"testing"

	contdriver "github.com/containerd/continuity/driver"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"
)

func verifyFile(t testing.TB, path string, mode os.FileMode, uid, gid uint32) {
	fi, err := os.Stat(path)
	require.NoError(t, err)

	actual := fi.Mode()
	assert.Equal(t, mode&os.ModeType, actual&os.ModeType, path)
	assert.Equal(t, mode&os.ModePerm, actual&os.ModePerm, path)
	assert.Equal(t, mode&os.ModeSticky, actual&os.ModeSticky, path)
	assert.Equal(t, mode&os.ModeSetuid, actual&os.ModeSetuid, path)
	assert.Equal(t, mode&os.ModeSetgid, actual&os.ModeSetgid, path)

	if stat, ok := fi.Sys().(*syscall.Stat_t); ok {
		assert.Equal(t, uid, stat.Uid, path)
		assert.Equal(t, gid, stat.Gid, path)
	}
}

func createBase(t testing.TB, driver graphdriver.Driver, name string) {
	// We need to be able to set any perms
	oldmask := unix.Umask(0)
	defer unix.Umask(oldmask)

	err := driver.CreateReadWrite(name, "", nil)
	require.NoError(t, err)

	dirFS, err := driver.Get(name, "")
	require.NoError(t, err)
	defer driver.Put(name)

	subdir := dirFS.Join(dirFS.Path(), "a subdir")
	require.NoError(t, dirFS.Mkdir(subdir, 0705|os.ModeSticky))
	require.NoError(t, dirFS.Lchown(subdir, 1, 2))

	file := dirFS.Join(dirFS.Path(), "a file")
	err = contdriver.WriteFile(dirFS, file, []byte("Some data"), 0222|os.ModeSetuid)
	require.NoError(t, err)
}

func verifyBase(t testing.TB, driver graphdriver.Driver, name string) {
	dirFS, err := driver.Get(name, "")
	require.NoError(t, err)
	defer driver.Put(name)

	subdir := dirFS.Join(dirFS.Path(), "a subdir")
	verifyFile(t, subdir, 0705|os.ModeDir|os.ModeSticky, 1, 2)

	file := dirFS.Join(dirFS.Path(), "a file")
	verifyFile(t, file, 0222|os.ModeSetuid, 0, 0)

	files, err := readDir(dirFS, dirFS.Path())
	require.NoError(t, err)
	assert.Len(t, files, 2)
}
