/*
Copyright 2016 The Kubernetes Authors.

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

package net

import (
	"net"
	"net/url"
	"os"
	"syscall"
	"testing"
)

func getIPNet(cidr string) *net.IPNet {
	_, ipnet, _ := net.ParseCIDR(cidr)
	return ipnet
}

func TestIPNetEqual(t *testing.T) {
	testCases := []struct {
		ipnet1 *net.IPNet
		ipnet2 *net.IPNet
		expect bool
	}{
		//null case
		{
			getIPNet("10.0.0.1/24"),
			getIPNet(""),
			false,
		},
		{
			getIPNet("10.0.0.0/24"),
			getIPNet("10.0.0.0/24"),
			true,
		},
		{
			getIPNet("10.0.0.0/24"),
			getIPNet("10.0.0.1/24"),
			true,
		},
		{
			getIPNet("10.0.0.0/25"),
			getIPNet("10.0.0.0/24"),
			false,
		},
		{
			getIPNet("10.0.1.0/24"),
			getIPNet("10.0.0.0/24"),
			false,
		},
	}

	for _, tc := range testCases {
		if tc.expect != IPNetEqual(tc.ipnet1, tc.ipnet2) {
			t.Errorf("Expect equality of %s and %s be to %v", tc.ipnet1.String(), tc.ipnet2.String(), tc.expect)
		}
	}
}

func TestIsConnectionRefused(t *testing.T) {
	testCases := []struct {
		err    error
		expect bool
	}{
		{
			&url.Error{Err: &net.OpError{Err: syscall.ECONNRESET}},
			false,
		},
		{
			&url.Error{Err: &net.OpError{Err: syscall.ECONNREFUSED}},
			true,
		},
		{&url.Error{Err: &net.OpError{Err: &os.SyscallError{Err: syscall.ECONNREFUSED}}},
			true,
		},
	}

	for _, tc := range testCases {
		if result := IsConnectionRefused(tc.err); result != tc.expect {
			t.Errorf("Expect to be %v, but actual is %v", tc.expect, result)
		}
	}
}
