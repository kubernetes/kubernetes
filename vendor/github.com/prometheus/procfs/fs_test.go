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

func TestFSXFSStats(t *testing.T) {
	stats, err := FS("fixtures").XFSStats()
	if err != nil {
		t.Fatalf("failed to parse XFS stats: %v", err)
	}

	// Very lightweight test just to sanity check the path used
	// to open XFS stats. Heavier tests in package xfs.
	if want, got := uint32(92447), stats.ExtentAllocation.ExtentsAllocated; want != got {
		t.Errorf("unexpected extents allocated:\nwant: %d\nhave: %d", want, got)
	}
}
