// Copyright The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build (aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris) && !js

package procfs

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"golang.org/x/sys/unix"
)

// ProcMapPermissions contains permission settings read from `/proc/[pid]/maps`.
type ProcMapPermissions struct {
	// mapping has the [R]ead flag set
	Read bool
	// mapping has the [W]rite flag set
	Write bool
	// mapping has the [X]ecutable flag set
	Execute bool
	// mapping has the [S]hared flag set
	Shared bool
	// mapping is marked as [P]rivate (copy on write)
	Private bool
}

// ProcMap contains the process memory-mappings of the process
// read from `/proc/[pid]/maps`.
type ProcMap struct {
	// The start address of current mapping.
	StartAddr uintptr
	// The end address of the current mapping
	EndAddr uintptr
	// The permissions for this mapping
	Perms *ProcMapPermissions
	// The current offset into the file/fd (e.g., shared libs)
	Offset int64
	// Device owner of this mapping (major:minor) in Mkdev format.
	Dev uint64
	// The inode of the device above
	Inode uint64
	// The file or psuedofile (or empty==anonymous)
	Pathname string
}

// parseDevice parses the device token of a line and converts it to a dev_t
// (mkdev) like structure.
func parseDevice(s string) (uint64, error) {
	i := strings.Index(s, ":")
	if i == -1 {
		return 0, fmt.Errorf("%w: expected separator `:` in %s", ErrFileParse, s)
	}

	major, err := strconv.ParseUint(s[0:i], 16, 0)
	if err != nil {
		return 0, err
	}

	minor, err := strconv.ParseUint(s[i+1:], 16, 0)
	if err != nil {
		return 0, err
	}

	return unix.Mkdev(uint32(major), uint32(minor)), nil
}

// parseAddress converts a hex-string to a uintptr.
func parseAddress(s string) (uintptr, error) {
	a, err := strconv.ParseUint(s, 16, 0)
	if err != nil {
		return 0, err
	}

	return uintptr(a), nil
}

// parseAddresses parses the start-end address.
func parseAddresses(s string) (uintptr, uintptr, error) {
	idx := strings.Index(s, "-")
	if idx == -1 {
		return 0, 0, fmt.Errorf("%w: expected separator `-` in %s", ErrFileParse, s)
	}

	saddr, err := parseAddress(s[0:idx])
	if err != nil {
		return 0, 0, err
	}

	eaddr, err := parseAddress(s[idx+1:])
	if err != nil {
		return 0, 0, err
	}

	return saddr, eaddr, nil
}

// parsePermissions parses a token and returns any that are set.
func parsePermissions(s string) (*ProcMapPermissions, error) {
	if len(s) < 4 {
		return nil, fmt.Errorf("%w: invalid permissions token", ErrFileParse)
	}

	perms := ProcMapPermissions{}
	for _, ch := range s {
		switch ch {
		case 'r':
			perms.Read = true
		case 'w':
			perms.Write = true
		case 'x':
			perms.Execute = true
		case 'p':
			perms.Private = true
		case 's':
			perms.Shared = true
		}
	}

	return &perms, nil
}

// parseProcMap will attempt to parse a single line within a proc/[pid]/maps
// buffer.
func parseProcMap(text string) (*ProcMap, error) {
	fields := strings.Fields(text)
	if len(fields) < 5 {
		return nil, fmt.Errorf("%w: truncated procmap entry", ErrFileParse)
	}

	saddr, eaddr, err := parseAddresses(fields[0])
	if err != nil {
		return nil, err
	}

	perms, err := parsePermissions(fields[1])
	if err != nil {
		return nil, err
	}

	offset, err := strconv.ParseInt(fields[2], 16, 0)
	if err != nil {
		return nil, err
	}

	device, err := parseDevice(fields[3])
	if err != nil {
		return nil, err
	}

	inode, err := strconv.ParseUint(fields[4], 10, 0)
	if err != nil {
		return nil, err
	}

	pathname := ""

	if len(fields) >= 5 {
		pathname = strings.Join(fields[5:], " ")
	}

	return &ProcMap{
		StartAddr: saddr,
		EndAddr:   eaddr,
		Perms:     perms,
		Offset:    offset,
		Dev:       device,
		Inode:     inode,
		Pathname:  pathname,
	}, nil
}

// ProcMaps reads from /proc/[pid]/maps to get the memory-mappings of the
// process.
func (p Proc) ProcMaps() ([]*ProcMap, error) {
	file, err := os.Open(p.path("maps"))
	if err != nil {
		return nil, err
	}
	defer file.Close()

	maps := []*ProcMap{}
	scan := bufio.NewScanner(file)

	for scan.Scan() {
		m, err := parseProcMap(scan.Text())
		if err != nil {
			return nil, err
		}

		maps = append(maps, m)
	}

	return maps, nil
}
