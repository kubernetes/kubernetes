// Copyright 2020 The Prometheus Authors
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
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
)

const (
	// Maximum size limit used by io.LimitReader while reading the content of the
	// /proc/net/udp{,6} files. The number of lines inside such a file is dynamic
	// as each line represents a single used socket.
	// In theory, the number of available sockets is 65535 (2^16 - 1) per IP.
	// With e.g. 150 Byte per line and the maximum number of 65535,
	// the reader needs to handle 150 Byte * 65535 =~ 10 MB for a single IP.
	readLimit = 4294967296 // Byte -> 4 GiB
)

// This contains generic data structures for both udp and tcp sockets.
type (
	// NetIPSocket represents the contents of /proc/net/{t,u}dp{,6} file without the header.
	NetIPSocket []*netIPSocketLine

	// NetIPSocketSummary provides already computed values like the total queue lengths or
	// the total number of used sockets. In contrast to NetIPSocket it does not collect
	// the parsed lines into a slice.
	NetIPSocketSummary struct {
		// TxQueueLength shows the total queue length of all parsed tx_queue lengths.
		TxQueueLength uint64
		// RxQueueLength shows the total queue length of all parsed rx_queue lengths.
		RxQueueLength uint64
		// UsedSockets shows the total number of parsed lines representing the
		// number of used sockets.
		UsedSockets uint64
		// Drops shows the total number of dropped packets of all UDP sockets.
		Drops *uint64
	}

	// A single line parser for fields from /proc/net/{t,u}dp{,6}.
	// Fields which are not used by IPSocket are skipped.
	// Drops is non-nil for udp{,6}, but nil for tcp{,6}.
	// For the proc file format details, see https://linux.die.net/man/5/proc.
	netIPSocketLine struct {
		Sl        uint64
		LocalAddr net.IP
		LocalPort uint64
		RemAddr   net.IP
		RemPort   uint64
		St        uint64
		TxQueue   uint64
		RxQueue   uint64
		UID       uint64
		Inode     uint64
		Drops     *uint64
	}
)

func newNetIPSocket(file string) (NetIPSocket, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var netIPSocket NetIPSocket
	isUDP := strings.Contains(file, "udp")

	lr := io.LimitReader(f, readLimit)
	s := bufio.NewScanner(lr)
	s.Scan() // skip first line with headers
	for s.Scan() {
		fields := strings.Fields(s.Text())
		line, err := parseNetIPSocketLine(fields, isUDP)
		if err != nil {
			return nil, err
		}
		netIPSocket = append(netIPSocket, line)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return netIPSocket, nil
}

// newNetIPSocketSummary creates a new NetIPSocket{,6} from the contents of the given file.
func newNetIPSocketSummary(file string) (*NetIPSocketSummary, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var netIPSocketSummary NetIPSocketSummary
	var udpPacketDrops uint64
	isUDP := strings.Contains(file, "udp")

	lr := io.LimitReader(f, readLimit)
	s := bufio.NewScanner(lr)
	s.Scan() // skip first line with headers
	for s.Scan() {
		fields := strings.Fields(s.Text())
		line, err := parseNetIPSocketLine(fields, isUDP)
		if err != nil {
			return nil, err
		}
		netIPSocketSummary.TxQueueLength += line.TxQueue
		netIPSocketSummary.RxQueueLength += line.RxQueue
		netIPSocketSummary.UsedSockets++
		if isUDP {
			udpPacketDrops += *line.Drops
			netIPSocketSummary.Drops = &udpPacketDrops
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	return &netIPSocketSummary, nil
}

// the /proc/net/{t,u}dp{,6} files are network byte order for ipv4 and for ipv6 the address is four words consisting of four bytes each. In each of those four words the four bytes are written in reverse order.

func parseIP(hexIP string) (net.IP, error) {
	var byteIP []byte
	byteIP, err := hex.DecodeString(hexIP)
	if err != nil {
		return nil, fmt.Errorf("%w: Cannot parse socket field in %q: %w", ErrFileParse, hexIP, err)
	}
	switch len(byteIP) {
	case 4:
		return net.IP{byteIP[3], byteIP[2], byteIP[1], byteIP[0]}, nil
	case 16:
		i := net.IP{
			byteIP[3], byteIP[2], byteIP[1], byteIP[0],
			byteIP[7], byteIP[6], byteIP[5], byteIP[4],
			byteIP[11], byteIP[10], byteIP[9], byteIP[8],
			byteIP[15], byteIP[14], byteIP[13], byteIP[12],
		}
		return i, nil
	default:
		return nil, fmt.Errorf("%w: Unable to parse IP %s: %v", ErrFileParse, hexIP, nil)
	}
}

// parseNetIPSocketLine parses a single line, represented by a list of fields.
func parseNetIPSocketLine(fields []string, isUDP bool) (*netIPSocketLine, error) {
	line := &netIPSocketLine{}
	if len(fields) < 10 {
		return nil, fmt.Errorf(
			"%w: Less than 10 columns found %q",
			ErrFileParse,
			strings.Join(fields, " "),
		)
	}
	var err error // parse error

	// sl
	s := strings.Split(fields[0], ":")
	if len(s) != 2 {
		return nil, fmt.Errorf("%w: Unable to parse sl field in line %q", ErrFileParse, fields[0])
	}

	if line.Sl, err = strconv.ParseUint(s[0], 0, 64); err != nil {
		return nil, fmt.Errorf("%w: Unable to parse sl field in %q: %w", ErrFileParse, line.Sl, err)
	}
	// local_address
	l := strings.Split(fields[1], ":")
	if len(l) != 2 {
		return nil, fmt.Errorf("%w: Unable to parse local_address field in %q", ErrFileParse, fields[1])
	}
	if line.LocalAddr, err = parseIP(l[0]); err != nil {
		return nil, err
	}
	if line.LocalPort, err = strconv.ParseUint(l[1], 16, 64); err != nil {
		return nil, fmt.Errorf("%w: Unable to parse local_address port value line %q: %w", ErrFileParse, line.LocalPort, err)
	}

	// remote_address
	r := strings.Split(fields[2], ":")
	if len(r) != 2 {
		return nil, fmt.Errorf("%w: Unable to parse rem_address field in %q", ErrFileParse, fields[1])
	}
	if line.RemAddr, err = parseIP(r[0]); err != nil {
		return nil, err
	}
	if line.RemPort, err = strconv.ParseUint(r[1], 16, 64); err != nil {
		return nil, fmt.Errorf("%w: Cannot parse rem_address port value in %q: %w", ErrFileParse, line.RemPort, err)
	}

	// st
	if line.St, err = strconv.ParseUint(fields[3], 16, 64); err != nil {
		return nil, fmt.Errorf("%w: Cannot parse st value in %q: %w", ErrFileParse, line.St, err)
	}

	// tx_queue and rx_queue
	q := strings.Split(fields[4], ":")
	if len(q) != 2 {
		return nil, fmt.Errorf(
			"%w: Missing colon for tx/rx queues in socket line %q",
			ErrFileParse,
			fields[4],
		)
	}
	if line.TxQueue, err = strconv.ParseUint(q[0], 16, 64); err != nil {
		return nil, fmt.Errorf("%w: Cannot parse tx_queue value in %q: %w", ErrFileParse, line.TxQueue, err)
	}
	if line.RxQueue, err = strconv.ParseUint(q[1], 16, 64); err != nil {
		return nil, fmt.Errorf("%w: Cannot parse trx_queue value in %q: %w", ErrFileParse, line.RxQueue, err)
	}

	// uid
	if line.UID, err = strconv.ParseUint(fields[7], 0, 64); err != nil {
		return nil, fmt.Errorf("%w: Cannot parse UID value in %q: %w", ErrFileParse, line.UID, err)
	}

	// inode
	if line.Inode, err = strconv.ParseUint(fields[9], 0, 64); err != nil {
		return nil, fmt.Errorf("%w: Cannot parse inode value in %q: %w", ErrFileParse, line.Inode, err)
	}

	// drops
	if isUDP {
		drops, err := strconv.ParseUint(fields[12], 0, 64)
		if err != nil {
			return nil, fmt.Errorf("%w: Cannot parse drops value in %q: %w", ErrFileParse, drops, err)
		}
		line.Drops = &drops
	}

	return line, nil
}
