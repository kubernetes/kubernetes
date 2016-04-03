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
	"reflect"
	"testing"
)

func TestNewLinuxCapabilitiesRetainSet(t *testing.T) {
	tests := []struct {
		in []string

		wset []LinuxCapability
		werr bool
	}{
		{
			[]string{},
			nil,
			true,
		},
		{
			[]string{"CAP_ADMIN"},
			[]LinuxCapability{"CAP_ADMIN"},
			false,
		},
		{
			[]string{"CAP_AUDIT_READ", "CAP_KILL"},
			[]LinuxCapability{"CAP_AUDIT_READ", "CAP_KILL"},
			false,
		},
	}
	for i, tt := range tests {
		c, err := NewLinuxCapabilitiesRetainSet(tt.in...)
		if tt.werr {
			if err == nil {
				t.Errorf("#%d: did not get expected error", i)
			}
			continue
		}
		if gset := c.Set(); !reflect.DeepEqual(gset, tt.wset) {
			t.Errorf("#%d: got %#v, want %#v", i, gset, tt.wset)
		}
	}

}

func TestNewLinuxCapabilitiesRevokeSet(t *testing.T) {
	tests := []struct {
		in []string

		wset []LinuxCapability
		werr bool
	}{
		{
			[]string{},
			[]LinuxCapability{},
			true,
		},
		{
			[]string{"CAP_AUDIT_WRITE"},
			[]LinuxCapability{"CAP_AUDIT_WRITE"},
			false,
		},
		{
			[]string{"CAP_SYS_ADMIN", "CAP_CHOWN"},
			[]LinuxCapability{"CAP_SYS_ADMIN", "CAP_CHOWN"},
			false,
		},
	}
	for i, tt := range tests {
		c, err := NewLinuxCapabilitiesRevokeSet(tt.in...)
		if tt.werr {
			if err == nil {
				t.Errorf("#%d: did not get expected error", i)
			}
			continue
		}
		if gset := c.Set(); !reflect.DeepEqual(gset, tt.wset) {
			t.Errorf("#%d: got %#v, want %#v", i, gset, tt.wset)
		}
	}

}
