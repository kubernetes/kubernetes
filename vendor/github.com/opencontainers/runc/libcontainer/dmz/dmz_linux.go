//go:build !runc_nodmz
// +build !runc_nodmz

package dmz

import (
	"bytes"
	"debug/elf"
	"embed"
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/sirupsen/logrus"
)

// Try to build the runc-dmz binary on "go generate". If it fails (or
// libcontainer is being imported as a library), the embed.FS will not contain
// the file, which will then cause us to fall back to a clone of
// /proc/self/exe.
//
// There is an empty file called dummy-file.txt in libcontainer/dmz/binary in
// order to work around the restriction that go:embed requires at least one
// file to match the pattern.
//
//go:generate make -B binary/runc-dmz
//go:embed binary
var runcDmzFs embed.FS

// A cached copy of the contents of runc-dmz.
var (
	runcDmzBinaryOnce    sync.Once
	runcDmzBinaryIsValid bool
	runcDmzBinary        []byte
)

// Binary returns a cloned copy (see CloneBinary) of a very minimal C program
// that just does an execve() of its arguments. This is used in the final
// execution step of the container execution as an intermediate process before
// the container process is execve'd. This allows for protection against
// CVE-2019-5736 without requiring a complete copy of the runc binary. Each
// call to Binary will return a new copy.
//
// If the runc-dmz binary is not embedded into the runc binary, Binary will
// return ErrNoDmzBinary as the error.
func Binary(tmpDir string) (*os.File, error) {
	// Only RUNC_DMZ=true enables runc_dmz.
	runcDmz := os.Getenv("RUNC_DMZ")
	if runcDmz == "" {
		logrus.Debugf("RUNC_DMZ is not set -- switching back to classic /proc/self/exe cloning")
		return nil, ErrNoDmzBinary
	}
	if dmzEnabled, err := strconv.ParseBool(runcDmz); err == nil && !dmzEnabled {
		logrus.Debugf("RUNC_DMZ is false -- switching back to classic /proc/self/exe cloning")
		return nil, ErrNoDmzBinary
	} else if err != nil {
		return nil, fmt.Errorf("parsing RUNC_DMZ: %w", err)
	}

	runcDmzBinaryOnce.Do(func() {
		runcDmzBinary, _ = runcDmzFs.ReadFile("binary/runc-dmz")
		// Verify that our embedded binary has a standard ELF header.
		if !bytes.HasPrefix(runcDmzBinary, []byte(elf.ELFMAG)) {
			if len(runcDmzBinary) != 0 {
				logrus.Infof("misconfigured build: embedded runc-dmz binary is non-empty but is missing a proper ELF header")
			}
		} else {
			runcDmzBinaryIsValid = true
		}
	})
	if !runcDmzBinaryIsValid {
		return nil, ErrNoDmzBinary
	}
	rdr := bytes.NewBuffer(runcDmzBinary)
	return CloneBinary(rdr, int64(rdr.Len()), "runc-dmz", tmpDir)
}
