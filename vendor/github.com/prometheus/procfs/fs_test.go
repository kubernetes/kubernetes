package procfs

import "testing"

func TestNewFS(t *testing.T) {
	if _, err := NewFS("foobar"); err == nil {
		t.Error("want NewFS to fail for non-existing mount point")
	}

	if _, err := NewFS("procfs.go"); err == nil {
		t.Error("want NewFS to fail if mount point is not a directory")
	}
}
