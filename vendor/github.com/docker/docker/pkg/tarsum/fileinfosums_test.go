package tarsum

import "testing"

func newFileInfoSums() FileInfoSums {
	return FileInfoSums{
		fileInfoSum{name: "file3", sum: "2abcdef1234567890", pos: 2},
		fileInfoSum{name: "dup1", sum: "deadbeef1", pos: 5},
		fileInfoSum{name: "file1", sum: "0abcdef1234567890", pos: 0},
		fileInfoSum{name: "file4", sum: "3abcdef1234567890", pos: 3},
		fileInfoSum{name: "dup1", sum: "deadbeef0", pos: 4},
		fileInfoSum{name: "file2", sum: "1abcdef1234567890", pos: 1},
	}
}

func TestSortFileInfoSums(t *testing.T) {
	dups := newFileInfoSums().GetAllFile("dup1")
	if len(dups) != 2 {
		t.Errorf("expected length 2, got %d", len(dups))
	}
	dups.SortByNames()
	if dups[0].Pos() != 4 {
		t.Errorf("sorted dups should be ordered by position. Expected 4, got %d", dups[0].Pos())
	}

	fis := newFileInfoSums()
	expected := "0abcdef1234567890"
	fis.SortBySums()
	got := fis[0].Sum()
	if got != expected {
		t.Errorf("Expected %q, got %q", expected, got)
	}

	fis = newFileInfoSums()
	expected = "dup1"
	fis.SortByNames()
	gotFis := fis[0]
	if gotFis.Name() != expected {
		t.Errorf("Expected %q, got %q", expected, gotFis.Name())
	}
	// since a duplicate is first, ensure it is ordered first by position too
	if gotFis.Pos() != 4 {
		t.Errorf("Expected %d, got %d", 4, gotFis.Pos())
	}

	fis = newFileInfoSums()
	fis.SortByPos()
	if fis[0].Pos() != 0 {
		t.Error("sorted fileInfoSums by Pos should order them by position.")
	}

	fis = newFileInfoSums()
	expected = "deadbeef1"
	gotFileInfoSum := fis.GetFile("dup1")
	if gotFileInfoSum.Sum() != expected {
		t.Errorf("Expected %q, got %q", expected, gotFileInfoSum)
	}
	if fis.GetFile("noPresent") != nil {
		t.Error("Should have return nil if name not found.")
	}

}
