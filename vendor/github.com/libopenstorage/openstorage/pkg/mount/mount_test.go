package mount

import (
	"fmt"
	"os"
	"syscall"
	"testing"

	"github.com/stretchr/testify/require"
)

const (
	source = "/tmp/ost/mount_test_src"
	dest   = "/tmp/ost/mount_test_dest"
)

var m Manager

func TestAll(t *testing.T) {
	setup(t)
	load(t)
	mountTest(t)
	inspect(t)
	hasMounts(t)
	refcounts(t)
	exists(t)
	shutdown(t)
}

func setup(t *testing.T) {
	var err error
	m, err = New(NFSMount, nil, []string{""}, nil, []string{})
	if err != nil {
		t.Fatalf("Failed to setup test %v", err)
	}
	cleandir(source)
	cleandir(dest)
}

func cleandir(dir string) {
	syscall.Unmount(dir, 0)
	os.RemoveAll(dir)
	os.MkdirAll(dir, 0755)
}

func load(t *testing.T) {
	require.NoError(t, m.Load([]string{""}), "Failed in load")
}

func mountTest(t *testing.T) {
	err := m.Mount(0, source, dest, "", syscall.MS_BIND, "", 0)
	require.NoError(t, err, "Failed in mount")
	err = m.Unmount(source, dest, 0, 0)
	require.NoError(t, err, "Failed in unmount")
}

func inspect(t *testing.T) {
	p := m.Inspect(source)
	require.Equal(t, 0, len(p), "Expect 0 mounts actual %v mounts", len(p))

	err := m.Mount(0, source, dest, "", syscall.MS_BIND, "", 0)
	require.NoError(t, err, "Failed in mount")
	p = m.Inspect(source)
	require.Equal(t, 1, len(p), "Expect 1 mounts actual %v mounts", len(p))
	require.Equal(t, dest, p[0].Path, "Expect %q got %q", dest, p[0].Path)
	s := m.GetSourcePaths()
	require.NotZero(t, 1, len(s), "Expect 1 source path, actual %v", s)
	err = m.Unmount(source, dest, 0, 0)
	require.NoError(t, err, "Failed in unmount")
}

func hasMounts(t *testing.T) {
	count := m.HasMounts(source)
	require.Equal(t, 0, count, "Expect 0 mounts actual %v mounts", count)

	mounts := 0
	for i := 0; i < 10; i++ {
		dir := fmt.Sprintf("%s%d", dest, i)
		cleandir(dir)
		err := m.Mount(0, source, dir, "", syscall.MS_BIND, "", 0)
		require.NoError(t, err, "Failed in mount")
		mounts++
		count = m.HasMounts(source)
		require.Equal(t, mounts, count, "Expect %v mounts actual %v mounts", mounts, count)
	}
	for i := 5; i >= 0; i-- {
		dir := fmt.Sprintf("%s%d", dest, i)
		err := m.Unmount(source, dir, 0, 0)
		require.NoError(t, err, "Failed in unmount")
		mounts--
		count = m.HasMounts(source)
		require.Equal(t, mounts, count, "Expect %v mounts actual %v mounts", mounts, count)
	}

	for i := 9; i > 5; i-- {
		dir := fmt.Sprintf("%s%d", dest, i)
		err := m.Unmount(source, dir, 0, 0)
		require.NoError(t, err, "Failed in mount")
		mounts--
		count = m.HasMounts(source)
		require.Equal(t, mounts, count, "Expect %v mounts actual %v mounts", mounts, count)
	}
	require.Equal(t, mounts, 0, "Expect 0 mounts actual %v mounts", mounts)
}

func refcounts(t *testing.T) {
	require.Equal(t, m.HasMounts(source) == 0, true, "Don't expect mounts in the beginning")
	for i := 0; i < 10; i++ {
		err := m.Mount(0, source, dest, "", syscall.MS_BIND, "", 0)
		require.NoError(t, err, "Failed in mount")
		require.Equal(t, m.HasMounts(source), 1, "Refcnt must be one")
	}

	err := m.Unmount(source, dest, 0, 0)
	require.NoError(t, err, "Failed in unmount")
	require.Equal(t, m.HasMounts(source), 0, "Refcnt must go down to zero")

	err = m.Unmount(source, dest, 0, 0)
	require.Error(t, err, "Unmount should fail")
}

func exists(t *testing.T) {
	err := m.Mount(0, source, dest, "", syscall.MS_BIND, "", 0)
	require.NoError(t, err, "Failed in mount")
	exists, _ := m.Exists(source, "foo")
	require.False(t, exists, "%q should not be mapped to foo", source)
	exists, _ = m.Exists(source, dest)
	require.True(t, exists, "%q should  be mapped to %q", source, dest)
	err = m.Unmount(source, dest, 0, 0)
	require.NoError(t, err, "Failed in unmount")
}

func shutdown(t *testing.T) {
	os.RemoveAll(dest)
	os.RemoveAll(source)
}
