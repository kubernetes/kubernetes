package fs

import (
	"fmt"

	"github.com/opencontainers/cgroups/fscommon"
)

type parseError = fscommon.ParseError

// malformedLine is used by all cgroupfs file parsers that expect a line
// in a particular format but get some garbage instead.
func malformedLine(path, file, line string) error {
	return &parseError{Path: path, File: file, Err: fmt.Errorf("malformed line: %s", line)}
}
