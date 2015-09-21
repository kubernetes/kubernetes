// +build linux

package xattr_test

import (
	"os"
	"testing"

	"github.com/docker/libcontainer/xattr"
)

func testXattr(t *testing.T) {
	tmp := "xattr_test"
	out, err := os.OpenFile(tmp, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal("failed")
	}
	attr := "user.test"
	out.Close()

	if !xattr.XattrEnabled(tmp) {
		t.Log("Disabled")
		t.Fatal("failed")
	}
	t.Log("Success")

	err = xattr.Setxattr(tmp, attr, "test")
	if err != nil {
		t.Fatal("failed")
	}

	var value string
	value, err = xattr.Getxattr(tmp, attr)
	if err != nil {
		t.Fatal("failed")
	}
	if value != "test" {
		t.Fatal("failed")
	}
	t.Log("Success")

	var names []string
	names, err = xattr.Listxattr(tmp)
	if err != nil {
		t.Fatal("failed")
	}

	var found int
	for _, name := range names {
		if name == attr {
			found = 1
		}
	}
	// Listxattr doesn't return trusted.* and system.* namespace
	// attrs when run in unprevileged mode.
	if found != 1 {
		t.Fatal("failed")
	}
	t.Log("Success")

	big := "0000000000000000000000000000000000000000000000000000000000000000000008c6419ad822dfe29283fb3ac98dcc5908810cb31f4cfe690040c42c144b7492eicompslf20dxmlpgz"
	// Test for long xattrs larger than 128 bytes
	err = xattr.Setxattr(tmp, attr, big)
	if err != nil {
		t.Fatal("failed to add long value")
	}
	value, err = xattr.Getxattr(tmp, attr)
	if err != nil {
		t.Fatal("failed to get long value")
	}
	t.Log("Success")

	if value != big {
		t.Fatal("failed, value doesn't match")
	}
	t.Log("Success")
}
