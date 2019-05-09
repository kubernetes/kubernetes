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

package flag

import (
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

func TestIPVar(t *testing.T) {
	defaultIP := "0.0.0.0"
	testCases := []struct {
		argc      string
		expectErr bool
		expectVal string
	}{

		{
			argc:      "blah --ip=1.2.3.4",
			expectVal: "1.2.3.4",
		},
		{
			argc:      "blah --ip=1.2.3.4a",
			expectErr: true,
			expectVal: defaultIP,
		},
	}
	for _, tc := range testCases {
		fs := pflag.NewFlagSet("blah", pflag.PanicOnError)
		ip := defaultIP
		fs.Var(IPVar{&ip}, "ip", "the ip")

		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = r.(error)
				}
			}()
			fs.Parse(strings.Split(tc.argc, " "))
		}()

		if tc.expectErr && err == nil {
			t.Errorf("did not observe an expected error")
			continue
		}
		if !tc.expectErr && err != nil {
			t.Errorf("observed an unexpected error: %v", err)
			continue
		}
		if tc.expectVal != ip {
			t.Errorf("unexpected ip: expected %q, saw %q", tc.expectVal, ip)
		}
	}
}

func TestIPPortVar(t *testing.T) {
	defaultIPPort := "0.0.0.0:8080"
	testCases := []struct {
		desc      string
		argc      string
		expectErr bool
		expectVal string
	}{

		{
			desc:      "valid ipv4 1",
			argc:      "blah --ipport=0.0.0.0",
			expectVal: "0.0.0.0",
		},
		{
			desc:      "valid ipv4 2",
			argc:      "blah --ipport=127.0.0.1",
			expectVal: "127.0.0.1",
		},

		{
			desc:      "invalid IP",
			argc:      "blah --ipport=invalidip",
			expectErr: true,
			expectVal: defaultIPPort,
		},
		{
			desc:      "valid ipv4 with port",
			argc:      "blah --ipport=0.0.0.0:8080",
			expectVal: "0.0.0.0:8080",
		},
		{
			desc:      "invalid ipv4 with invalid port",
			argc:      "blah --ipport=0.0.0.0:invalidport",
			expectErr: true,
			expectVal: defaultIPPort,
		},
		{
			desc:      "invalid IP with port",
			argc:      "blah --ipport=invalidip:8080",
			expectErr: true,
			expectVal: defaultIPPort,
		},
		{
			desc:      "valid ipv6 1",
			argc:      "blah --ipport=::1",
			expectVal: "::1",
		},
		{
			desc:      "valid ipv6 2",
			argc:      "blah --ipport=::",
			expectVal: "::",
		},
		{
			desc:      "valid ipv6 with port",
			argc:      "blah --ipport=[::1]:8080",
			expectVal: "[::1]:8080",
		},
		{
			desc:      "invalid ipv6 with port without bracket",
			argc:      "blah --ipport=fd00:f00d:600d:f00d:8080",
			expectErr: true,
			expectVal: defaultIPPort,
		},
	}
	for _, tc := range testCases {
		fs := pflag.NewFlagSet("blah", pflag.PanicOnError)
		ipport := defaultIPPort
		fs.Var(IPPortVar{&ipport}, "ipport", "the ip:port")

		var err error
		func() {
			defer func() {
				if r := recover(); r != nil {
					err = r.(error)
				}
			}()
			fs.Parse(strings.Split(tc.argc, " "))
		}()

		if tc.expectErr && err == nil {
			t.Errorf("%q: Did not observe an expected error", tc.desc)
			continue
		}
		if !tc.expectErr && err != nil {
			t.Errorf("%q: Observed an unexpected error: %v", tc.desc, err)
			continue
		}
		if tc.expectVal != ipport {
			t.Errorf("%q: Unexpected ipport: expected %q, saw %q", tc.desc, tc.expectVal, ipport)
		}
	}
}
