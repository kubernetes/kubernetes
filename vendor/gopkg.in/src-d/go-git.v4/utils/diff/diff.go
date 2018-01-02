// Package diff implements line oriented diffs, similar to the ancient
// Unix diff command.
//
// The current implementation is just a wrapper around Sergi's
// go-diff/diffmatchpatch library, which is a go port of Neil
// Fraser's google-diff-match-patch code
package diff

import (
	"bytes"

	"github.com/sergi/go-diff/diffmatchpatch"
)

// Do computes the (line oriented) modifications needed to turn the src
// string into the dst string.
func Do(src, dst string) (diffs []diffmatchpatch.Diff) {
	dmp := diffmatchpatch.New()
	wSrc, wDst, warray := dmp.DiffLinesToChars(src, dst)
	diffs = dmp.DiffMain(wSrc, wDst, false)
	diffs = dmp.DiffCharsToLines(diffs, warray)
	return diffs
}

// Dst computes and returns the destination text.
func Dst(diffs []diffmatchpatch.Diff) string {
	var text bytes.Buffer
	for _, d := range diffs {
		if d.Type != diffmatchpatch.DiffDelete {
			text.WriteString(d.Text)
		}
	}
	return text.String()
}

// Src computes and returns the source text
func Src(diffs []diffmatchpatch.Diff) string {
	var text bytes.Buffer
	for _, d := range diffs {
		if d.Type != diffmatchpatch.DiffInsert {
			text.WriteString(d.Text)
		}
	}
	return text.String()
}
