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
	// readLimit is used by io.LimitReader while reading the content of the
	// /proc/net/udp{,6} files. The number of lines inside such a file is dynamic
	// as each line represents a single used socket.
	// In theory, the number of available sockets is 65535 (2^16 - 1) per IP.
	// With e.g. 150 Byte per line and the maximum number of 65535,
	// the reader needs to handle 150 Byte * 65535 =~ 10 MB for a single IP.
	readLimit = 4294967296 // Byte -> 4 GiB
)

// this contains generic data structures for both udp and tcp sockets
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
	}

	// netIPSocketLine represents the fields parsed from a single line
	// in /proc/net/{t,u}dp{,6}. Fields which are not used by IPSocket are skipped.
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
	}
)

func newNetIPSocket(file string) (NetIPSocket, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var netIPSocket NetIPSocket

	lr := io.LimitReader(f, readLimit)
	s := bufio.NewScanner(lr)
	s.Scan() // skip first line with headers
	for s.Scan() {
		fields := strings.Fields(s.Text())
		line, err := parseNetIPSocketLine(fields)
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

	lr := io.LimitReader(f, readLimit)
	s := bufio.NewScanner(lr)
	s.Scan() // skip first line with headers
	for s.Scan() {
		fields := strings.Fields(s.Text())
		line, err := parseNetIPSocketLine(fields)
		if err != nil {
			return nil, err
		}
		netIPSocketSummary.TxQueueLength += line.TxQueue
		netIPSocketSummary.RxQueueLength += line.RxQueue
		netIPSocketSummary.UsedSockets++
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
		return nil, fmt.Errorf("cannot parse address field in socket line %q", hexIP)
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
		return nil, fmt.Errorf("Unable to parse IP %s", hexIP)
	}
}

// parseNetIPSocketLine parses a single line, represented by a list of fields.
func parseNetIPSocketLine(fields []string) (*netIPSocketLine, error) {
	line := &netIPSocketLine{}
	if len(fields) < 8 {
		return nil, fmt.Errorf(
			"cannot parse net socket line as it has less then 8 columns %q",
			strings.Join(fields, " "),
		)
	}
	var err error // parse error

	// sl
	s := strings.Split(fields[0], ":")
	if len(s) != 2 {
		return nil, fmt.Errorf("cannot parse sl field in socket line %q", fields[0])
	}

	if line.Sl, err = strconv.ParseUint(s[0], 0, 64); err != nil {
		return nil, fmt.Errorf("cannot parse sl value in socket line: %w", err)
	}
	// local_address
	l := strings.Split(fields[1], ":")
	if len(l) != 2 {
		return nil, fmt.Errorf("cannot parse local_address field in socket line %q", fields[1])
	}
	if line.LocalAddr, err = parseIP(l[0]); err != nil {
		return nil, err
	}
	if line.LocalPort, err = strconv.ParseUint(l[1], 16, 64); err != nil {
		return nil, fmt.Errorf("cannot parse local_address port value in socket line: %w", err)
	}

	// remote_address
	r := strings.Split(fields[2], ":")
	if len(r) != 2 {
		return nil, fmt.Errorf("cannot parse rem_address field in socket line %q", fields[1])
	}
	if line.RemAddr, err = parseIP(r[0]); err != nil {
		return nil, err
	}
	if line.RemPort, err = strconv.ParseUint(r[1], 16, 64); err != nil {
		return nil, fmt.Errorf("cannot parse rem_address port value in socket line: %w", err)
	}

	// st
	if line.St, err = strconv.ParseUint(fields[3], 16, 64); err != nil {
		return nil, fmt.Errorf("cannot parse st value in socket line: %w", err)
	}

	// tx_queue and rx_queue
	q := strings.Split(fields[4], ":")
	if len(q) != 2 {
		return nil, fmt.Errorf(
			"cannot parse tx/rx queues in socket line as it has a missing colon %q",
			fields[4],
		)
	}
	if line.TxQueue, err = strconv.ParseUint(q[0], 16, 64); err != nil {
		return nil, fmt.Errorf("cannot parse tx_queue value in socket line: %w", err)
	}
	if line.RxQueue, err = strconv.ParseUint(q[1], 16, 64); err != nil {
		return nil, fmt.Errorf("cannot parse rx_queue value in socket line: %w", err)
	}

	// uid
	if line.UID, err = strconv.ParseUint(fields[7], 0, 64); err != nil {
		return nil, fmt.Errorf("cannot parse uid value in socket line: %w", err)
	}

	return line, nil
}
