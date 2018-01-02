// +build linux

package fs

import (
	"io/ioutil"
	"os"
	"os/exec"
	"testing"

	"github.com/containerd/containerd/testutil"
	"github.com/stretchr/testify/assert"
)

func testSupportsDType(t *testing.T, expected bool, mkfs ...string) {
	testutil.RequiresRoot(t)
	mnt, err := ioutil.TempDir("", "containerd-fs-test-supports-dtype")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(mnt)

	deviceName, cleanupDevice, err := testutil.NewLoopback(100 << 20) // 100 MB
	if err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(mkfs[0], append(mkfs[1:], deviceName)...).CombinedOutput(); err != nil {
		// not fatal
		t.Skipf("could not mkfs (%v) %s: %v (out: %q)", mkfs, deviceName, err, string(out))
	}
	if out, err := exec.Command("mount", deviceName, mnt).CombinedOutput(); err != nil {
		// not fatal
		t.Skipf("could not mount %s: %v (out: %q)", deviceName, err, string(out))
	}
	defer func() {
		testutil.Unmount(t, mnt)
		cleanupDevice()
	}()
	// check whether it supports d_type
	result, err := SupportsDType(mnt)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Supports d_type: %v", result)
	assert.Equal(t, expected, result)
}

func TestSupportsDTypeWithFType0XFS(t *testing.T) {
	testSupportsDType(t, false, "mkfs.xfs", "-m", "crc=0", "-n", "ftype=0")
}

func TestSupportsDTypeWithFType1XFS(t *testing.T) {
	testSupportsDType(t, true, "mkfs.xfs", "-m", "crc=0", "-n", "ftype=1")
}

func TestSupportsDTypeWithExt4(t *testing.T) {
	testSupportsDType(t, true, "mkfs.ext4", "-F")
}
