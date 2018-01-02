// +build linux

// FIXME: we can't put this test to the mount package:
// import cycle not allowed in test
// package github.com/containerd/containerd/mount (test)
//         imports github.com/containerd/containerd/testutil
//         imports github.com/containerd/containerd/mount
//
// NOTE: we can't have this as lookup_test (compilation fails)
package lookuptest

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/containerd/containerd/mount"
	"github.com/containerd/containerd/testutil"
	"github.com/stretchr/testify/assert"
)

func testLookup(t *testing.T, fsType string) {
	checkLookup := func(mntPoint, dir string) {
		info, err := mount.Lookup(dir)
		if err != nil {
			t.Fatal(err)
		}
		assert.Equal(t, fsType, info.FSType)
		assert.Equal(t, mntPoint, info.Mountpoint)
	}

	testutil.RequiresRoot(t)
	mnt, err := ioutil.TempDir("", "containerd-mountinfo-test-lookup")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(mnt)

	deviceName, cleanupDevice, err := testutil.NewLoopback(100 << 20) // 100 MB
	if err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("mkfs", "-t", fsType, deviceName).CombinedOutput(); err != nil {
		// not fatal
		t.Skipf("could not mkfs (%s) %s: %v (out: %q)", fsType, deviceName, err, string(out))
	}
	if out, err := exec.Command("mount", deviceName, mnt).CombinedOutput(); err != nil {
		// not fatal
		t.Skipf("could not mount %s: %v (out: %q)", deviceName, err, string(out))
	}
	defer func() {
		testutil.Unmount(t, mnt)
		cleanupDevice()
	}()
	assert.True(t, strings.HasPrefix(deviceName, "/dev/loop"))
	checkLookup(mnt, mnt)

	newMnt, err := ioutil.TempDir("", "containerd-mountinfo-test-newMnt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(newMnt)

	if out, err := exec.Command("mount", "--bind", mnt, newMnt).CombinedOutput(); err != nil {
		t.Fatalf("could not mount %s to %s: %v (out: %q)", mnt, newMnt, err, string(out))
	}
	defer func() {
		testutil.Unmount(t, newMnt)
	}()
	checkLookup(newMnt, newMnt)

	subDir := filepath.Join(newMnt, "subDir")
	err = os.MkdirAll(subDir, 0700)
	if err != nil {
		t.Fatal(err)
	}
	checkLookup(newMnt, subDir)
}

func TestLookupWithExt4(t *testing.T) {
	testLookup(t, "ext4")
}

func TestLookupWithXFS(t *testing.T) {
	testLookup(t, "xfs")
}
