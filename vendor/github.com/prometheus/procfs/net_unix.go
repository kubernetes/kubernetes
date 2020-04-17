// Copyright 2018 The Prometheus Authors
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

package procfs

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// For the proc file format details,
// see https://elixir.bootlin.com/linux/v4.17/source/net/unix/af_unix.c#L2815
// and https://elixir.bootlin.com/linux/latest/source/include/uapi/linux/net.h#L48.

const (
	netUnixKernelPtrIdx = iota
	netUnixRefCountIdx
	_
	netUnixFlagsIdx
	netUnixTypeIdx
	netUnixStateIdx
	netUnixInodeIdx

	// Inode and Path are optional.
	netUnixStaticFieldsCnt = 6
)

const (
	netUnixTypeStream    = 1
	netUnixTypeDgram     = 2
	netUnixTypeSeqpacket = 5

	netUnixFlagListen = 1 << 16

	netUnixStateUnconnected  = 1
	netUnixStateConnecting   = 2
	netUnixStateConnected    = 3
	netUnixStateDisconnected = 4
)

var errInvalidKernelPtrFmt = errors.New("Invalid Num(the kernel table slot number) format")

// NetUnixType is the type of the type field.
type NetUnixType uint64

// NetUnixFlags is the type of the flags field.
type NetUnixFlags uint64

// NetUnixState is the type of the state field.
type NetUnixState uint64

// NetUnixLine represents a line of /proc/net/unix.
type NetUnixLine struct {
	KernelPtr string
	RefCount  uint64
	Protocol  uint64
	Flags     NetUnixFlags
	Type      NetUnixType
	State     NetUnixState
	Inode     uint64
	Path      string
}

// NetUnix holds the data read from /proc/net/unix.
type NetUnix struct {
	Rows []*NetUnixLine
}

// NewNetUnix returns data read from /proc/net/unix.
func NewNetUnix() (*NetUnix, error) {
	fs, err := NewFS(DefaultMountPoint)
	if err != nil {
		return nil, err
	}

	return fs.NewNetUnix()
}

// NewNetUnix returns data read from /proc/net/unix.
func (fs FS) NewNetUnix() (*NetUnix, error) {
	return NewNetUnixByPath(fs.proc.Path("net/unix"))
}

// NewNetUnixByPath returns data read from /proc/net/unix by file path.
// It might returns an error with partial parsed data, if an error occur after some data parsed.
func NewNetUnixByPath(path string) (*NetUnix, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return NewNetUnixByReader(f)
}

// NewNetUnixByReader returns data read from /proc/net/unix by a reader.
// It might returns an error with partial parsed data, if an error occur after some data parsed.
func NewNetUnixByReader(reader io.Reader) (*NetUnix, error) {
	nu := &NetUnix{
		Rows: make([]*NetUnixLine, 0, 32),
	}
	scanner := bufio.NewScanner(reader)
	// Omit the header line.
	scanner.Scan()
	header := scanner.Text()
	// From the man page of proc(5), it does not contain an Inode field,
	// but in actually it exists.
	// This code works for both cases.
	hasInode := strings.Contains(header, "Inode")

	minFieldsCnt := netUnixStaticFieldsCnt
	if hasInode {
		minFieldsCnt++
	}
	for scanner.Scan() {
		line := scanner.Text()
		item, err := nu.parseLine(line, hasInode, minFieldsCnt)
		if err != nil {
			return nu, err
		}
		nu.Rows = append(nu.Rows, item)
	}

	return nu, scanner.Err()
}

func (u *NetUnix) parseLine(line string, hasInode bool, minFieldsCnt int) (*NetUnixLine, error) {
	fields := strings.Fields(line)
	fieldsLen := len(fields)
	if fieldsLen < minFieldsCnt {
		return nil, fmt.Errorf(
			"Parse Unix domain failed: expect at least %d fields but got %d",
			minFieldsCnt, fieldsLen)
	}
	kernelPtr, err := u.parseKernelPtr(fields[netUnixKernelPtrIdx])
	if err != nil {
		return nil, fmt.Errorf("Parse Unix domain num(%s) failed: %s", fields[netUnixKernelPtrIdx], err)
	}
	users, err := u.parseUsers(fields[netUnixRefCountIdx])
	if err != nil {
		return nil, fmt.Errorf("Parse Unix domain ref count(%s) failed: %s", fields[netUnixRefCountIdx], err)
	}
	flags, err := u.parseFlags(fields[netUnixFlagsIdx])
	if err != nil {
		return nil, fmt.Errorf("Parse Unix domain flags(%s) failed: %s", fields[netUnixFlagsIdx], err)
	}
	typ, err := u.parseType(fields[netUnixTypeIdx])
	if err != nil {
		return nil, fmt.Errorf("Parse Unix domain type(%s) failed: %s", fields[netUnixTypeIdx], err)
	}
	state, err := u.parseState(fields[netUnixStateIdx])
	if err != nil {
		return nil, fmt.Errorf("Parse Unix domain state(%s) failed: %s", fields[netUnixStateIdx], err)
	}
	var inode uint64
	if hasInode {
		inodeStr := fields[netUnixInodeIdx]
		inode, err = u.parseInode(inodeStr)
		if err != nil {
			return nil, fmt.Errorf("Parse Unix domain inode(%s) failed: %s", inodeStr, err)
		}
	}

	nuLine := &NetUnixLine{
		KernelPtr: kernelPtr,
		RefCount:  users,
		Type:      typ,
		Flags:     flags,
		State:     state,
		Inode:     inode,
	}

	// Path field is optional.
	if fieldsLen > minFieldsCnt {
		pathIdx := netUnixInodeIdx + 1
		if !hasInode {
			pathIdx--
		}
		nuLine.Path = fields[pathIdx]
	}

	return nuLine, nil
}

func (u NetUnix) parseKernelPtr(str string) (string, error) {
	if !strings.HasSuffix(str, ":") {
		return "", errInvalidKernelPtrFmt
	}
	return str[:len(str)-1], nil
}

func (u NetUnix) parseUsers(hexStr string) (uint64, error) {
	return strconv.ParseUint(hexStr, 16, 32)
}

func (u NetUnix) parseProtocol(hexStr string) (uint64, error) {
	return strconv.ParseUint(hexStr, 16, 32)
}

func (u NetUnix) parseType(hexStr string) (NetUnixType, error) {
	typ, err := strconv.ParseUint(hexStr, 16, 16)
	if err != nil {
		return 0, err
	}
	return NetUnixType(typ), nil
}

func (u NetUnix) parseFlags(hexStr string) (NetUnixFlags, error) {
	flags, err := strconv.ParseUint(hexStr, 16, 32)
	if err != nil {
		return 0, err
	}
	return NetUnixFlags(flags), nil
}

func (u NetUnix) parseState(hexStr string) (NetUnixState, error) {
	st, err := strconv.ParseInt(hexStr, 16, 8)
	if err != nil {
		return 0, err
	}
	return NetUnixState(st), nil
}

func (u NetUnix) parseInode(inodeStr string) (uint64, error) {
	return strconv.ParseUint(inodeStr, 10, 64)
}

func (t NetUnixType) String() string {
	switch t {
	case netUnixTypeStream:
		return "stream"
	case netUnixTypeDgram:
		return "dgram"
	case netUnixTypeSeqpacket:
		return "seqpacket"
	}
	return "unknown"
}

func (f NetUnixFlags) String() string {
	switch f {
	case netUnixFlagListen:
		return "listen"
	default:
		return "default"
	}
}

func (s NetUnixState) String() string {
	switch s {
	case netUnixStateUnconnected:
		return "unconnected"
	case netUnixStateConnecting:
		return "connecting"
	case netUnixStateConnected:
		return "connected"
	case netUnixStateDisconnected:
		return "disconnected"
	}
	return "unknown"
}
