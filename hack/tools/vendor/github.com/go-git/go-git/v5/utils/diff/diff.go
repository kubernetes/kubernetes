// Package diff implements line oriented diffs, similar to the ancient
// Unix diff command.
//
// The current implementation is just a wrapper around Sergi's
// go-diff/diffmatchpatch library, which is a go port of Neil
// Fraser's google-diff-match-patch code
package diff

import (
	"bytes"
	"time"

	"github.com/sergi/go-diff/diffmatchpatch"
)

// Do computes the (line oriented) modifications needed to turn the src
// string into the dst string. The underlying algorithm is Meyers,
// its complexity is O(N*d) where N is min(lines(src), lines(dst)) and d
// is the size of the diff.
func Do(src, dst string) (diffs []diffmatchpatch.Diff) {
	// the default timeout is time.Second which may be too small under heavy load
	return DoWithTimeout(src, dst, time.Hour)
}

// DoWithTimeout computes the (line oriented) modifications needed to turn the src
// string into the dst string. The `timeout` argument specifies the maximum
// amount of time it is allowed to spend in this function. If the timeout
// is exceeded, the parts of the strings which were not considered are turned into
// a bulk delete+insert and the half-baked suboptimal result is returned at once.
// The underlying algorithm is Meyers, its complexity is O(N*d) where N is
// min(lines(src), lines(dst)) and d is the size of the diff.
func DoWithTimeout(src, dst string, timeout time.Duration) (diffs []diffmatchpatch.Diff) {
	dmp := diffmatchpatch.New()
	dmp.DiffTimeout = timeout
	wSrc, wDst, warray := dmp.DiffLinesToRunes(src, dst)
	diffs = dmp.DiffMainRunes(wSrc, wDst, false)
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
