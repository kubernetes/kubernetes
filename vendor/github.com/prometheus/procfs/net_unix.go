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
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// For the proc file format details,
// see https://elixir.bootlin.com/linux/v4.17/source/net/unix/af_unix.c#L2815
// and https://elixir.bootlin.com/linux/latest/source/include/uapi/linux/net.h#L48.

// Constants for the various /proc/net/unix enumerations.
// TODO: match against x/sys/unix or similar?
const (
	netUnixTypeStream    = 1
	netUnixTypeDgram     = 2
	netUnixTypeSeqpacket = 5

	netUnixFlagDefault = 0
	netUnixFlagListen  = 1 << 16

	netUnixStateUnconnected  = 1
	netUnixStateConnecting   = 2
	netUnixStateConnected    = 3
	netUnixStateDisconnected = 4
)

// NetUNIXType is the type of the type field.
type NetUNIXType uint64

// NetUNIXFlags is the type of the flags field.
type NetUNIXFlags uint64

// NetUNIXState is the type of the state field.
type NetUNIXState uint64

// NetUNIXLine represents a line of /proc/net/unix.
type NetUNIXLine struct {
	KernelPtr string
	RefCount  uint64
	Protocol  uint64
	Flags     NetUNIXFlags
	Type      NetUNIXType
	State     NetUNIXState
	Inode     uint64
	Path      string
}

// NetUNIX holds the data read from /proc/net/unix.
type NetUNIX struct {
	Rows []*NetUNIXLine
}

// NetUNIX returns data read from /proc/net/unix.
func (fs FS) NetUNIX() (*NetUNIX, error) {
	return readNetUNIX(fs.proc.Path("net/unix"))
}

// readNetUNIX reads data in /proc/net/unix format from the specified file.
func readNetUNIX(file string) (*NetUNIX, error) {
	// This file could be quite large and a streaming read is desirable versus
	// reading the entire contents at once.
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseNetUNIX(f)
}

// parseNetUNIX creates a NetUnix structure from the incoming stream.
func parseNetUNIX(r io.Reader) (*NetUNIX, error) {
	// Begin scanning by checking for the existence of Inode.
	s := bufio.NewScanner(r)
	s.Scan()

	// From the man page of proc(5), it does not contain an Inode field,
	// but in actually it exists. This code works for both cases.
	hasInode := strings.Contains(s.Text(), "Inode")

	// Expect a minimum number of fields, but Inode and Path are optional:
	// Num       RefCount Protocol Flags    Type St Inode Path
	minFields := 6
	if hasInode {
		minFields++
	}

	var nu NetUNIX
	for s.Scan() {
		line := s.Text()
		item, err := nu.parseLine(line, hasInode, minFields)
		if err != nil {
			return nil, fmt.Errorf("%w: /proc/net/unix encountered data %q: %w", ErrFileParse, line, err)
		}

		nu.Rows = append(nu.Rows, item)
	}

	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("%w: /proc/net/unix encountered data: %w", ErrFileParse, err)
	}

	return &nu, nil
}

func (u *NetUNIX) parseLine(line string, hasInode bool, min int) (*NetUNIXLine, error) {
	fields := strings.Fields(line)

	l := len(fields)
	if l < min {
		return nil, fmt.Errorf("%w: expected at least %d fields but got %d", ErrFileParse, min, l)
	}

	// Field offsets are as follows:
	// Num       RefCount Protocol Flags    Type St Inode Path

	kernelPtr := strings.TrimSuffix(fields[0], ":")

	users, err := u.parseUsers(fields[1])
	if err != nil {
		return nil, fmt.Errorf("%w: ref count %q: %w", ErrFileParse, fields[1], err)
	}

	flags, err := u.parseFlags(fields[3])
	if err != nil {
		return nil, fmt.Errorf("%w: Unable to parse flags %q: %w", ErrFileParse, fields[3], err)
	}

	typ, err := u.parseType(fields[4])
	if err != nil {
		return nil, fmt.Errorf("%w: Failed to parse type %q: %w", ErrFileParse, fields[4], err)
	}

	state, err := u.parseState(fields[5])
	if err != nil {
		return nil, fmt.Errorf("%w: Failed to parse state %q: %w", ErrFileParse, fields[5], err)
	}

	var inode uint64
	if hasInode {
		inode, err = u.parseInode(fields[6])
		if err != nil {
			return nil, fmt.Errorf("%w failed to parse inode %q: %w", ErrFileParse, fields[6], err)
		}
	}

	n := &NetUNIXLine{
		KernelPtr: kernelPtr,
		RefCount:  users,
		Type:      typ,
		Flags:     flags,
		State:     state,
		Inode:     inode,
	}

	// Path field is optional.
	if l > min {
		// Path occurs at either index 6 or 7 depending on whether inode is
		// already present.
		pathIdx := 7
		if !hasInode {
			pathIdx--
		}

		n.Path = fields[pathIdx]
	}

	return n, nil
}

func (u NetUNIX) parseUsers(s string) (uint64, error) {
	return strconv.ParseUint(s, 16, 32)
}

func (u NetUNIX) parseType(s string) (NetUNIXType, error) {
	typ, err := strconv.ParseUint(s, 16, 16)
	if err != nil {
		return 0, err
	}

	return NetUNIXType(typ), nil
}

func (u NetUNIX) parseFlags(s string) (NetUNIXFlags, error) {
	flags, err := strconv.ParseUint(s, 16, 32)
	if err != nil {
		return 0, err
	}

	return NetUNIXFlags(flags), nil
}

func (u NetUNIX) parseState(s string) (NetUNIXState, error) {
	st, err := strconv.ParseInt(s, 16, 8)
	if err != nil {
		return 0, err
	}

	return NetUNIXState(st), nil
}

func (u NetUNIX) parseInode(s string) (uint64, error) {
	return strconv.ParseUint(s, 10, 64)
}

func (t NetUNIXType) String() string {
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

func (f NetUNIXFlags) String() string {
	switch f {
	case netUnixFlagListen:
		return "listen"
	default:
		return "default"
	}
}

func (s NetUNIXState) String() string {
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
