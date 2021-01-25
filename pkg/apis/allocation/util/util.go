/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package util

import (
	"math/big"
	"net"
)

// IPToDecimal converts an IP to a string with its decimal representation
func IPToDecimal(ip net.IP) string {
	if ip == nil {
		return ""
	}
	i := new(big.Int)
	// compare the address with its decimal representation
	if ip.To4() != nil {
		i.SetBytes(ip.To4())
	} else if ip.To16() != nil {
		i.SetBytes(ip.To16())
	}
	return i.String()
}

// DecimalToIP converts a string with the decimal representation of an IP address and returns the IP
func DecimalToIP(ipDecimal string) net.IP {
	if len(ipDecimal) == 0 {
		return nil
	}
	// convert decimal representation to BigInt
	i, ok := new(big.Int).SetString(ipDecimal, 10)
	// if we can not convert the string to a number
	// or the number has more than 16 bytes it is not an IP
	if !ok || len(i.Bytes()) > 16 {
		return nil
	}
	bufLen := net.IPv4len
	if len(i.Bytes()) > net.IPv4len {
		bufLen = net.IPv6len
	}
	// convert BigInt to IP address, truncate to IP Len bytes
	r := append(make([]byte, bufLen), i.Bytes()...)
	return net.IP(r[len(r)-bufLen:])
}
