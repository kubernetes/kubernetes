// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package flags

import (
	"testing"
)

func TestIPAddressPortSet(t *testing.T) {
	pass := []string{
		"1.2.3.4:8080",
		"10.1.1.1:80",
		"[2001:db8::1]:8080",
	}

	fail := []string{
		// bad IP specification
		":2379",
		"127.0:8080",
		"123:456",
		// bad port specification
		"127.0.0.1:foo",
		"127.0.0.1:",
		// unix sockets not supported
		"unix://",
		"unix://tmp/etcd.sock",
		// bad strings
		"somewhere",
		"234#$",
		"file://foo/bar",
		"http://hello",
		"2001:db8::1",
		"2001:db8::1:1",
	}

	for i, tt := range pass {
		f := &IPAddressPort{}
		if err := f.Set(tt); err != nil {
			t.Errorf("#%d: unexpected error from IPAddressPort.Set(%q): %v", i, tt, err)
		}
	}

	for i, tt := range fail {
		f := &IPAddressPort{}
		if err := f.Set(tt); err == nil {
			t.Errorf("#%d: expected error from IPAddressPort.Set(%q)", i, tt)
		}
	}
}

func TestIPAddressPortString(t *testing.T) {
	addresses := []string{
		"[2001:db8::1:1234]:2379",
		"127.0.0.1:2379",
	}
	for i, tt := range addresses {
		f := &IPAddressPort{}
		if err := f.Set(tt); err != nil {
			t.Errorf("#%d: unexpected error: %v", i, err)
		}

		want := tt
		got := f.String()
		if want != got {
			t.Errorf("#%d: IPAddressPort.String() value should be %q, got %q", i, want, got)
		}
	}
}
