// This package implements the diff.Differ interface using github.com/mb0/diff as a backend.
package difflib

import (
	"io"
	"io/ioutil"

	"github.com/pmezard/go-difflib/difflib"
)

type differ struct{}

// New returns an implementation of diff.Differ using mb0diff as the backend.
func New() *differ {
	return &differ{}
}

// Diff consumes the entire reader streams into memory before generating a diff
// which then gets filled into the buffer. This implementation stores and
// manipulates all three values in memory.
func (diff *differ) Diff(out io.Writer, a io.ReadSeeker, b io.ReadSeeker) error {
	var src, dst []byte
	var err error

	if src, err = ioutil.ReadAll(a); err != nil {
		return err
	}
	if dst, err = ioutil.ReadAll(b); err != nil {
		return err
	}

	d := difflib.UnifiedDiff{
		A:       difflib.SplitLines(string(src)),
		B:       difflib.SplitLines(string(dst)),
		Context: 3,
	}

	return difflib.WriteUnifiedDiff(out, d)
}
