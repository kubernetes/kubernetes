// Copyright 2015 The appc Authors
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

package types

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestIsolatorUnmarshal(t *testing.T) {
	tests := []struct {
		msg  string
		werr bool
	}{
		{
			`{
				"name": "os/linux/cpu-shares",
				"value": 2048
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/cpu-shares",
				"value": 1
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/cpu-shares",
				"value": "pants"
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/cpu-shares",
				"value": "262145"
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/cpu-shares",
				"value": "-1"
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/oom-score-adj",
				"value": 250
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/oom-score-adj",
				"value": -250
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/oom-score-adj",
				"value": -2500
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/oom-score-adj",
				"value": "pants"
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/no-new-privileges",
				"value": true
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/no-new-privileges",
				"value": false
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/no-new-privileges",
				"value": 123
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/no-new-privileges",
				"value": {"set": ["CAP_KILL"]}
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/no-new-privileges",
				"value": "foo"
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/capabilities-retain-set",
				"value": {"set": ["CAP_KILL"]}
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/capabilities-retain-set",
				"value": {"set": ["CAP_PONIES"]}
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/capabilities-retain-set",
				"value": {"set": []}
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/capabilities-retain-set",
				"value": {"set": "CAP_PONIES"}
			}`,
			true,
		},
		{
			`{
				"name": "resource/block-bandwidth",
				"value": {"default": true, "limit": "1G"}
			}`,
			false,
		},
		{
			`{
				"name": "resource/block-bandwidth",
				"value": {"default": false, "limit": "1G"}
			}`,
			true,
		},
		{
			`{
				"name": "resource/block-bandwidth",
				"value": {"request": "30G", "limit": "1G"}
			}`,
			true,
		},
		{
			`{
				"name": "resource/block-iops",
				"value": {"default": true, "limit": "1G"}
			}`,
			false,
		},
		{
			`{
				"name": "resource/block-iops",
				"value": {"default": false, "limit": "1G"}
			}`,
			true,
		},
		{
			`{
				"name": "resource/block-iops",
				"value": {"request": "30G", "limit": "1G"}
			}`,
			true,
		},
		{
			`{
				"name": "resource/cpu",
				"value": {"request": "30m", "limit": "1m"}
			}`,
			false,
		},
		{
			`{
				"name": "resource/memory",
				"value": {"request": "1G", "limit": "2Gi"}
			}`,
			false,
		},
		{
			`{
				"name": "resource/memory",
				"value": {"default": true, "request": "1G", "limit": "2G"}
			}`,
			true,
		},
		{
			`{
				"name": "resource/network-bandwidth",
				"value": {"default": true, "limit": "1G"}
			}`,
			false,
		},
		{
			`{
				"name": "resource/network-bandwidth",
				"value": {"default": false, "limit": "1G"}
			}`,
			true,
		},
		{
			`{
				"name": "resource/network-bandwidth",
				"value": {"request": "30G", "limit": "1G"}
			}`,
			true,
		},
		{
			`{
				"name": "os/unix/sysctl",
				"value": {"net.ipv4.tcp_rfc1337": "1"}
			}`,
			false,
		},
		{
			`{
				"name": "os/unix/sysctl",
				"value": {"net.ipv4.tcp_rfc1337": 1}
			}`,
			true,
		},
		{
			`{
				"name": "os/unix/sysctl",
				"value": {["net.ipv4.tcp_rfc1337"]}
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/selinux-context",
				"value": {"user" : "user_u", "role": "role_r", "type": "type_r", "level": "s0-s0"}
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/selinux-context",
				"value": {"user" : "user_u", "role": "role_r", "type": "type_r", "level": "s0-s0:c0"}
			}`,
			false,
		},
		{
			`{
				"name": "os/linux/selinux-context",
				"value": {"user" : "user_u", "role": "role_r:type_t", "type": "type_r", "level": "s0-s0"}
			}`,
			true,
		},
		{
			`{
				"name": "os/linux/selinux-context",
				"value": {"user" : "user_u", "role": "", "type": "type_r", "level": "s0-s0"}
			}`,
			true,
		},
	}

	for i, tt := range tests {
		var ii Isolator
		err := ii.UnmarshalJSON([]byte(tt.msg))
		if gerr := (err != nil); gerr != tt.werr {
			t.Errorf("#%d: gerr=%t, want %t (err=%v)", i, gerr, tt.werr, err)
		}
	}
}

func TestIsolatorsGetByName(t *testing.T) {
	ex := `
		[
			{
				"name": "resource/cpu",
				"value": {"request": "30m", "limit": "1m"}
			},
			{
				"name": "resource/memory",
				"value": {"request": "1G", "limit": "2Gi"}
			},
			{
				"name": "os/linux/capabilities-retain-set",
				"value": {"set": ["CAP_KILL"]}
			},
			{
				"name": "os/linux/capabilities-remove-set",
				"value": {"set": ["CAP_KILL"]}
			},
			{
				"name": "os/linux/no-new-privileges",
				"value": true
			},
			{
				"name": "os/unix/sysctl",
				"value": {"net.ipv4.tcp_rfc1337": "1"}
			}
		]
	`

	tests := []struct {
		name     ACIdentifier
		wlimit   int64
		wrequest int64
		wset     []LinuxCapability
		wprivs   LinuxNoNewPrivileges
		wsysctl  UnixSysctl
	}{
		{"resource/cpu", 1, 30, nil, false, nil},
		{"resource/memory", 2147483648, 1000000000, nil, false, nil},
		{"os/linux/capabilities-retain-set", 0, 0, []LinuxCapability{"CAP_KILL"}, false, nil},
		{"os/linux/capabilities-remove-set", 0, 0, []LinuxCapability{"CAP_KILL"}, false, nil},
		{"os/linux/no-new-privileges", 0, 0, nil, LinuxNoNewPrivileges(true), nil},
		{"os/unix/sysctl", 0, 0, nil, false, UnixSysctl{"net.ipv4.tcp_rfc1337": "1"}},
	}

	var is Isolators
	err := json.Unmarshal([]byte(ex), &is)
	if err != nil {
		panic(err)
	}

	if len(is) < 2 {
		t.Fatalf("too few items %v", len(is))
	}

	for i, tt := range tests {
		c := is.GetByName(tt.name)
		if c == nil {
			t.Fatalf("can't find item %v in %v items", tt.name, len(is))
		}
		switch v := c.Value().(type) {
		case Resource:
			var r Resource = v
			glimit := r.Limit()
			grequest := r.Request()

			var vlimit, vrequest int64
			if tt.name == "resource/cpu" {
				vlimit, vrequest = glimit.MilliValue(), grequest.MilliValue()
			} else {
				vlimit, vrequest = glimit.Value(), grequest.Value()
			}

			if vlimit != tt.wlimit {
				t.Errorf("#%d: glimit=%v, want %v", i, vlimit, tt.wlimit)
			}
			if vrequest != tt.wrequest {
				t.Errorf("#%d: grequest=%v, want %v", i, vrequest, tt.wrequest)
			}

		case LinuxCapabilitiesSet:
			var s LinuxCapabilitiesSet = v
			if !reflect.DeepEqual(s.Set(), tt.wset) {
				t.Errorf("#%d: gset=%v, want %v", i, s.Set(), tt.wset)
			}

		case *LinuxNoNewPrivileges:
			if tt.wprivs != *v {
				t.Errorf("#%d: got %v, want %v", i, v, tt.wprivs)
			}

		case *UnixSysctl:
			if !reflect.DeepEqual(*v, tt.wsysctl) {
				t.Errorf("#%d: got %v, want %v", i, *v, tt.wsysctl)
			}

		default:
			panic("unexpected type")
		}
	}
}

func TestIsolatorUnrecognized(t *testing.T) {
	msg := `
		[{
			"name": "resource/network-bandwidth",
			"value": {"default": true, "limit": "1G"}
		},
		{
			"name": "resource/network-ponies",
			"value": 0
		}]`

	ex := Isolators{
		{Name: "resource/network-ponies"},
	}

	is := Isolators{}
	if err := json.Unmarshal([]byte(msg), &is); err != nil {
		t.Fatalf("failed to unmarshal isolators: %v", err)
	}

	u := is.Unrecognized()
	if len(u) != len(ex) {
		t.Errorf("unrecognized isolator list is wrong len: want %v, got %v", len(ex), len(u))
	}

	for i, e := range ex {
		if e.Name != u[i].Name {
			t.Errorf("unrecognized isolator list mismatch: want %v, got %v", e.Name, u[i].Name)
		}
	}
}
