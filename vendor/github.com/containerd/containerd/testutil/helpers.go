package testutil

import (
	"flag"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

var rootEnabled bool

func init() {
	flag.BoolVar(&rootEnabled, "test.root", false, "enable tests that require root")
}

// DumpDir will log out all of the contents of the provided directory to
// testing logger.
//
// Use this in a defer statement within tests that may allocate and exercise a
// temporary directory. Immensely useful for sanity checking and debugging
// failing tests.
//
// One should still test that contents are as expected. This is only a visual
// tool to assist when things don't go your way.
func DumpDir(t *testing.T, root string) {
	if err := filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if fi.Mode()&os.ModeSymlink != 0 {
			target, err := os.Readlink(path)
			if err != nil {
				return err
			}
			t.Log(fi.Mode(), path, "->", target)
		} else if fi.Mode().IsRegular() {
			p, err := ioutil.ReadFile(path)
			if err != nil {
				t.Logf("error reading file: %v", err)
				return nil
			}

			if len(p) > 64 { // just display a little bit.
				p = p[:64]
			}

			t.Log(fi.Mode(), path, "[", strconv.Quote(string(p)), "...]")
		} else {
			t.Log(fi.Mode(), path)
		}

		return nil
	}); err != nil {
		t.Fatalf("error dumping directory: %v", err)
	}
}
