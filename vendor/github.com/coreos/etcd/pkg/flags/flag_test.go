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
	"flag"
	"net/url"
	"os"
	"reflect"
	"testing"

	"github.com/coreos/etcd/pkg/transport"
)

func TestSetFlagsFromEnv(t *testing.T) {
	fs := flag.NewFlagSet("testing", flag.ExitOnError)
	fs.String("a", "", "")
	fs.String("b", "", "")
	fs.String("c", "", "")
	fs.Parse([]string{})

	os.Clearenv()
	// flags should be settable using env vars
	os.Setenv("ETCD_A", "foo")
	// and command-line flags
	if err := fs.Set("b", "bar"); err != nil {
		t.Fatal(err)
	}
	// command-line flags take precedence over env vars
	os.Setenv("ETCD_C", "woof")
	if err := fs.Set("c", "quack"); err != nil {
		t.Fatal(err)
	}

	// first verify that flags are as expected before reading the env
	for f, want := range map[string]string{
		"a": "",
		"b": "bar",
		"c": "quack",
	} {
		if got := fs.Lookup(f).Value.String(); got != want {
			t.Fatalf("flag %q=%q, want %q", f, got, want)
		}
	}

	// now read the env and verify flags were updated as expected
	err := SetFlagsFromEnv(fs)
	if err != nil {
		t.Errorf("err=%v, want nil", err)
	}
	for f, want := range map[string]string{
		"a": "foo",
		"b": "bar",
		"c": "quack",
	} {
		if got := fs.Lookup(f).Value.String(); got != want {
			t.Errorf("flag %q=%q, want %q", f, got, want)
		}
	}
}

func TestSetFlagsFromEnvBad(t *testing.T) {
	// now verify that an error is propagated
	fs := flag.NewFlagSet("testing", flag.ExitOnError)
	fs.Int("x", 0, "")
	os.Setenv("ETCD_X", "not_a_number")
	if err := SetFlagsFromEnv(fs); err == nil {
		t.Errorf("err=nil, want != nil")
	}
}

func TestSetBindAddrFromAddr(t *testing.T) {
	tests := []struct {
		args  []string
		waddr *IPAddressPort
	}{
		// no flags set
		{
			args:  []string{},
			waddr: &IPAddressPort{},
		},
		// addr flag set
		{
			args:  []string{"-addr=192.0.3.17:2379"},
			waddr: &IPAddressPort{IP: "::", Port: 2379},
		},
		// bindAddr flag set
		{
			args:  []string{"-bind-addr=127.0.0.1:2379"},
			waddr: &IPAddressPort{IP: "127.0.0.1", Port: 2379},
		},
		// both addr flags set
		{
			args:  []string{"-bind-addr=127.0.0.1:2379", "-addr=192.0.3.17:2379"},
			waddr: &IPAddressPort{IP: "127.0.0.1", Port: 2379},
		},
		// both addr flags set, IPv6
		{
			args:  []string{"-bind-addr=[2001:db8::4:9]:2379", "-addr=[2001:db8::4:f0]:2379"},
			waddr: &IPAddressPort{IP: "2001:db8::4:9", Port: 2379},
		},
	}
	for i, tt := range tests {
		fs := flag.NewFlagSet("test", flag.PanicOnError)
		fs.Var(&IPAddressPort{}, "addr", "")
		bindAddr := &IPAddressPort{}
		fs.Var(bindAddr, "bind-addr", "")
		if err := fs.Parse(tt.args); err != nil {
			t.Errorf("#%d: failed to parse flags: %v", i, err)
			continue
		}
		SetBindAddrFromAddr(fs, "bind-addr", "addr")

		if !reflect.DeepEqual(bindAddr, tt.waddr) {
			t.Errorf("#%d: bindAddr = %+v, want %+v", i, bindAddr, tt.waddr)
		}
	}
}

func TestURLsFromFlags(t *testing.T) {
	tests := []struct {
		args     []string
		tlsInfo  transport.TLSInfo
		wantURLs []url.URL
		wantFail bool
	}{
		// use -urls default when no flags defined
		{
			args:    []string{},
			tlsInfo: transport.TLSInfo{},
			wantURLs: []url.URL{
				{Scheme: "http", Host: "127.0.0.1:2379"},
			},
			wantFail: false,
		},

		// explicitly setting -urls should carry through
		{
			args:    []string{"-urls=https://192.0.3.17:2930,http://127.0.0.1:1024"},
			tlsInfo: transport.TLSInfo{},
			wantURLs: []url.URL{
				{Scheme: "http", Host: "127.0.0.1:1024"},
				{Scheme: "https", Host: "192.0.3.17:2930"},
			},
			wantFail: false,
		},

		// explicitly setting -addr should carry through
		{
			args:    []string{"-addr=192.0.2.3:1024"},
			tlsInfo: transport.TLSInfo{},
			wantURLs: []url.URL{
				{Scheme: "http", Host: "192.0.2.3:1024"},
			},
			wantFail: false,
		},

		// scheme prepended to -addr should be https if TLSInfo non-empty
		{
			args: []string{"-addr=192.0.2.3:1024"},
			tlsInfo: transport.TLSInfo{
				CertFile: "/tmp/foo",
				KeyFile:  "/tmp/bar",
			},
			wantURLs: []url.URL{
				{Scheme: "https", Host: "192.0.2.3:1024"},
			},
			wantFail: false,
		},

		// explicitly setting both -urls and -addr should fail
		{
			args:     []string{"-urls=https://127.0.0.1:1024", "-addr=192.0.2.3:1024"},
			tlsInfo:  transport.TLSInfo{},
			wantURLs: nil,
			wantFail: true,
		},
	}

	for i, tt := range tests {
		fs := flag.NewFlagSet("test", flag.PanicOnError)
		fs.Var(NewURLsValue("http://127.0.0.1:2379"), "urls", "")
		fs.Var(&IPAddressPort{}, "addr", "")

		if err := fs.Parse(tt.args); err != nil {
			t.Errorf("#%d: failed to parse flags: %v", i, err)
			continue
		}

		gotURLs, err := URLsFromFlags(fs, "urls", "addr", tt.tlsInfo)
		if tt.wantFail != (err != nil) {
			t.Errorf("#%d: wantFail=%t, got err=%v", i, tt.wantFail, err)
			continue
		}

		if !reflect.DeepEqual(tt.wantURLs, gotURLs) {
			t.Errorf("#%d: incorrect URLs\nwant=%#v\ngot=%#v", i, tt.wantURLs, gotURLs)
		}
	}
}
