package storage

import (
	"fmt"
	"sort"
	"testing"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/storage/driver"
	"github.com/docker/distribution/registry/storage/driver/inmemory"
)

func testFS(t *testing.T) (driver.StorageDriver, map[string]string, context.Context) {
	d := inmemory.New()
	ctx := context.Background()

	expected := map[string]string{
		"/a":       "dir",
		"/a/b":     "dir",
		"/a/b/c":   "dir",
		"/a/b/c/d": "file",
		"/a/b/c/e": "file",
		"/a/b/f":   "dir",
		"/a/b/f/g": "file",
		"/a/b/f/h": "file",
		"/a/b/f/i": "file",
		"/z":       "dir",
		"/z/y":     "file",
	}

	for p, typ := range expected {
		if typ != "file" {
			continue
		}

		if err := d.PutContent(ctx, p, []byte(p)); err != nil {
			t.Fatalf("unable to put content into fixture: %v", err)
		}
	}

	return d, expected, ctx
}

func TestWalkErrors(t *testing.T) {
	d, expected, ctx := testFS(t)
	fileCount := len(expected)
	err := Walk(ctx, d, "", func(fileInfo driver.FileInfo) error {
		return nil
	})
	if err == nil {
		t.Error("Expected invalid root err")
	}

	errEarlyExpected := fmt.Errorf("Early termination")

	err = Walk(ctx, d, "/", func(fileInfo driver.FileInfo) error {
		// error on the 2nd file
		if fileInfo.Path() == "/a/b" {
			return errEarlyExpected
		}

		delete(expected, fileInfo.Path())
		return nil
	})
	if len(expected) != fileCount-1 {
		t.Error("Walk failed to terminate with error")
	}
	if err != errEarlyExpected {
		if err == nil {
			t.Fatalf("expected an error due to early termination")
		} else {
			t.Error(err.Error())
		}
	}

	err = Walk(ctx, d, "/nonexistent", func(fileInfo driver.FileInfo) error {
		return nil
	})
	if err == nil {
		t.Errorf("Expected missing file err")
	}

}

func TestWalk(t *testing.T) {
	d, expected, ctx := testFS(t)
	var traversed []string
	err := Walk(ctx, d, "/", func(fileInfo driver.FileInfo) error {
		filePath := fileInfo.Path()
		filetype, ok := expected[filePath]
		if !ok {
			t.Fatalf("Unexpected file in walk: %q", filePath)
		}

		if fileInfo.IsDir() {
			if filetype != "dir" {
				t.Errorf("Unexpected file type: %q", filePath)
			}
		} else {
			if filetype != "file" {
				t.Errorf("Unexpected file type: %q", filePath)
			}

			// each file has its own path as the contents. If the length
			// doesn't match the path length, fail.
			if fileInfo.Size() != int64(len(fileInfo.Path())) {
				t.Fatalf("unexpected size for %q: %v != %v",
					fileInfo.Path(), fileInfo.Size(), len(fileInfo.Path()))
			}
		}
		delete(expected, filePath)
		traversed = append(traversed, filePath)
		return nil
	})
	if len(expected) > 0 {
		t.Errorf("Missed files in walk: %q", expected)
	}

	if !sort.StringsAreSorted(traversed) {
		t.Errorf("result should be sorted: %v", traversed)
	}

	if err != nil {
		t.Fatalf(err.Error())
	}
}

func TestWalkSkipDir(t *testing.T) {
	d, expected, ctx := testFS(t)
	err := Walk(ctx, d, "/", func(fileInfo driver.FileInfo) error {
		filePath := fileInfo.Path()
		if filePath == "/a/b" {
			// skip processing /a/b/c and /a/b/c/d
			return ErrSkipDir
		}
		delete(expected, filePath)
		return nil
	})
	if err != nil {
		t.Fatalf(err.Error())
	}
	if _, ok := expected["/a/b/c"]; !ok {
		t.Errorf("/a/b/c not skipped")
	}
	if _, ok := expected["/a/b/c/d"]; !ok {
		t.Errorf("/a/b/c/d not skipped")
	}
	if _, ok := expected["/a/b/c/e"]; !ok {
		t.Errorf("/a/b/c/e not skipped")
	}

}
