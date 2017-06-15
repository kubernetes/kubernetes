// Copyright (c) 2016 Uber Technologies, Inc.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package utils

import (
	"encoding/binary"
	"errors"
	"net"
	"strconv"
	"strings"
	"time"
)

var (
	// ErrEmptyIP an error for empty ip strings
	ErrEmptyIP = errors.New("empty string given for ip")

	// ErrNotHostColonPort an error for invalid host port string
	ErrNotHostColonPort = errors.New("expecting host:port")

	// ErrNotFourOctets an error for the wrong number of octets after splitting a string
	ErrNotFourOctets = errors.New("Wrong number of octets")
)

// ParseIPToUint32 converts a string ip (e.g. "x.y.z.w") to an uint32
func ParseIPToUint32(ip string) (uint32, error) {
	if ip == "" {
		return 0, ErrEmptyIP
	}

	if ip == "localhost" {
		return 127<<24 | 1, nil
	}

	octets := strings.Split(ip, ".")
	if len(octets) != 4 {
		return 0, ErrNotFourOctets
	}

	var intIP uint32
	for i := 0; i < 4; i++ {
		octet, err := strconv.Atoi(octets[i])
		if err != nil {
			return 0, err
		}
		intIP = (intIP << 8) | uint32(octet)
	}

	return intIP, nil
}

// ParsePort converts port number from string to uin16
func ParsePort(portString string) (uint16, error) {
	port, err := strconv.ParseUint(portString, 10, 16)
	return uint16(port), err
}

// PackIPAsUint32 packs an IPv4 as uint32
func PackIPAsUint32(ip net.IP) uint32 {
	if ipv4 := ip.To4(); ipv4 != nil {
		return binary.BigEndian.Uint32(ipv4)
	}
	return 0
}

// TimeToMicrosecondsSinceEpochInt64 converts Go time.Time to a long
// representing time since epoch in microseconds, which is used expected
// in the Jaeger spans encoded as Thrift.
func TimeToMicrosecondsSinceEpochInt64(t time.Time) int64 {
	// ^^^ Passing time.Time by value is faster than passing a pointer!
	// BenchmarkTimeByValue-8	2000000000	         1.37 ns/op
	// BenchmarkTimeByPtr-8  	2000000000	         1.98 ns/op

	return t.UnixNano() / 1000
}
