// +build linux freebsd

package graphtest

import (
	"io/ioutil"
	"os"
	"path"
	"syscall"
	"testing"

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

	dir, err := driver.Get(name, "")
	require.NoError(t, err)
	defer driver.Put(name)

	subdir := path.Join(dir, "a subdir")
	require.NoError(t, os.Mkdir(subdir, 0705|os.ModeSticky))
	require.NoError(t, os.Chown(subdir, 1, 2))

	file := path.Join(dir, "a file")
	err = ioutil.WriteFile(file, []byte("Some data"), 0222|os.ModeSetuid)
	require.NoError(t, err)
}

func verifyBase(t testing.TB, driver graphdriver.Driver, name string) {
	dir, err := driver.Get(name, "")
	require.NoError(t, err)
	defer driver.Put(name)

	subdir := path.Join(dir, "a subdir")
	verifyFile(t, subdir, 0705|os.ModeDir|os.ModeSticky, 1, 2)

	file := path.Join(dir, "a file")
	verifyFile(t, file, 0222|os.ModeSetuid, 0, 0)

	files, err := readDir(dir)
	require.NoError(t, err)
	assert.Len(t, files, 2)
}
