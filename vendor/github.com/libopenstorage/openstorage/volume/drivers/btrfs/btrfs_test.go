// +build linux,have_btrfs

package btrfs

import (
	"os"
	"os/exec"
	"testing"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/volume/drivers/test"
)

const (
	btrfsFile = "/var/btrfs"
	testPath  = "/var/test_dir"

	KiB = 1024
	MiB = KiB * 1024
	GiB = MiB * 1024
)

func TestAll(t *testing.T) {
	output, err := exec.Command("umount", btrfsFile).Output()
	if err != nil {
		t.Logf("error on umount %s (not fatal): %s %v", btrfsFile, string(output), err)
	}
	if err := os.Remove(btrfsFile); err != nil {
		t.Logf("error on rm %s (not fatal): %v", btrfsFile, err)
	}
	if err := os.MkdirAll(testPath, 0755); err != nil {
		t.Fatalf("failed on mkdir -p %s: %v", testPath, err)
	}
	file, err := os.Create(btrfsFile)
	if err != nil {
		t.Fatalf("failed to setup btrfs file at %s: %v", btrfsFile, err)
	}
	if err := file.Truncate(GiB); err != nil {
		t.Fatalf("failed to truncate %s 1G  %v", btrfsFile, err)
	}
	output, err = exec.Command("mkfs", "-t", "btrfs", "-f", btrfsFile).Output()
	if err != nil {
		t.Fatalf("failed to format to btrfs: %s %v", string(output), err)
	}
	output, err = exec.Command("mount", btrfsFile, testPath).Output()
	if err != nil {
		t.Fatalf("failed to mount to btrfs: %s %v", string(output), err)
	}
	volumeDriver, err := Init(map[string]string{RootParam: testPath})
	if err != nil {
		t.Fatalf("failed to initialize Driver: %v", err)
	}
	ctx := test.NewContext(volumeDriver)
	ctx.Filesystem = api.FSType_FS_TYPE_BTRFS
	test.Run(t, ctx)
}
