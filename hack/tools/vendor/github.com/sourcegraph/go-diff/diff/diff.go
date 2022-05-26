package diff

import (
	"bytes"
	"time"
)

// A FileDiff represents a unified diff for a single file.
//
// A file unified diff has a header that resembles the following:
//
//   --- oldname	2009-10-11 15:12:20.000000000 -0700
//   +++ newname	2009-10-11 15:12:30.000000000 -0700
type FileDiff struct {
	// the original name of the file
	OrigName string
	// the original timestamp (nil if not present)
	OrigTime *time.Time
	// the new name of the file (often same as OrigName)
	NewName string
	// the new timestamp (nil if not present)
	NewTime *time.Time
	// extended header lines (e.g., git's "new mode <mode>", "rename from <path>", etc.)
	Extended []string
	// hunks that were changed from orig to new
	Hunks []*Hunk
}

// A Hunk represents a series of changes (additions or deletions) in a file's
// unified diff.
type Hunk struct {
	// starting line number in original file
	OrigStartLine int32
	// number of lines the hunk applies to in the original file
	OrigLines int32
	// if > 0, then the original file had a 'No newline at end of file' mark at this offset
	OrigNoNewlineAt int32
	// starting line number in new file
	NewStartLine int32
	// number of lines the hunk applies to in the new file
	NewLines int32
	// optional section heading
	Section string
	// 0-indexed line offset in unified file diff (including section headers); this is
	// only set when Hunks are read from entire file diff (i.e., when ReadAllHunks is
	// called) This accounts for hunk headers, too, so the StartPosition of the first
	// hunk will be 1.
	StartPosition int32
	// hunk body (lines prefixed with '-', '+', or ' ')
	Body []byte
}

// A Stat is a diff stat that represents the number of lines added/changed/deleted.
type Stat struct {
	// number of lines added
	Added int32
	// number of lines changed
	Changed int32
	// number of lines deleted
	Deleted int32
}

// Stat computes the number of lines added/changed/deleted in all
// hunks in this file's diff.
func (d *FileDiff) Stat() Stat {
	total := Stat{}
	for _, h := range d.Hunks {
		total.add(h.Stat())
	}
	return total
}

// Stat computes the number of lines added/changed/deleted in this
// hunk.
func (h *Hunk) Stat() Stat {
	lines := bytes.Split(h.Body, []byte{'\n'})
	var last byte
	st := Stat{}
	for _, line := range lines {
		if len(line) == 0 {
			last = 0
			continue
		}
		switch line[0] {
		case '-':
			if last == '+' {
				st.Added--
				st.Changed++
				last = 0 // next line can't change this one since this is already a change
			} else {
				st.Deleted++
				last = line[0]
			}
		case '+':
			if last == '-' {
				st.Deleted--
				st.Changed++
				last = 0 // next line can't change this one since this is already a change
			} else {
				st.Added++
				last = line[0]
			}
		default:
			last = 0
		}
	}
	return st
}

var (
	hunkPrefix          = []byte("@@ ")
	onlyInMessagePrefix = []byte("Only in ")
)

const hunkHeader = "@@ -%d,%d +%d,%d @@"
const onlyInMessage = "Only in %s: %s\n"

// diffTimeParseLayout is the layout used to parse the time in unified diff file
// header timestamps.
// See https://www.gnu.org/software/diffutils/manual/html_node/Detailed-Unified.html.
const diffTimeParseLayout = "2006-01-02 15:04:05 -0700"

// diffTimeFormatLayout is the layout used to format (i.e., print) the time in unified diff file
// header timestamps.
// See https://www.gnu.org/software/diffutils/manual/html_node/Detailed-Unified.html.
const diffTimeFormatLayout = "2006-01-02 15:04:05.000000000 -0700"

func (s *Stat) add(o Stat) {
	s.Added += o.Added
	s.Changed += o.Changed
	s.Deleted += o.Deleted
}
