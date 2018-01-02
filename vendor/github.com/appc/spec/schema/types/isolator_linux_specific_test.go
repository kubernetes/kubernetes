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

func TestNewLinuxSELinuxContext(t *testing.T) {
	tests := []struct {
		inUser  string
		inRole  string
		inType  string
		inLevel string

		wuser  LinuxSELinuxUser
		wrole  LinuxSELinuxRole
		wtype  LinuxSELinuxType
		wlevel LinuxSELinuxLevel
		werr   bool
	}{
		{
			"unconfined_u", "object_r", "user_home_t", "s0",
			LinuxSELinuxUser("unconfined_u"),
			LinuxSELinuxRole("object_r"),
			LinuxSELinuxType("user_home_t"),
			LinuxSELinuxLevel("s0"),
			false,
		},
		{
			"unconfined_u", "object_r", "user_home_t", "s0-s0:c0",
			LinuxSELinuxUser("unconfined_u"),
			LinuxSELinuxRole("object_r"),
			LinuxSELinuxType("user_home_t"),
			LinuxSELinuxLevel("s0-s0:c0"),
			false,
		},
		{
			"", "object_r", "user_home_t", "s0",
			"",
			"",
			"",
			"",
			true,
		},
		{
			"unconfined_u:unconfined_t", "object_r", "user_home_t", "s0",
			"",
			"",
			"",
			"",
			true,
		},
	}
	for i, tt := range tests {
		c, err := NewLinuxSELinuxContext(tt.inUser, tt.inRole, tt.inType, tt.inLevel)
		if tt.werr {
			if err == nil {
				t.Errorf("#%d: did not get expected error", i)
			}
			continue
		}
		if guser := c.User(); !reflect.DeepEqual(guser, tt.wuser) {
			t.Errorf("#%d: got user %#v, want user %#v", i, guser, tt.wuser)
		}
		if grole := c.Role(); !reflect.DeepEqual(grole, tt.wrole) {
			t.Errorf("#%d: got role %#v, want role %#v", i, grole, tt.wrole)
		}
		if gtype := c.Type(); !reflect.DeepEqual(gtype, tt.wtype) {
			t.Errorf("#%d: got type %#v, want type %#v", i, gtype, tt.wtype)
		}
		if glevel := c.Level(); !reflect.DeepEqual(glevel, tt.wlevel) {
			t.Errorf("#%d: got level %#v, want level %#v", i, glevel, tt.wlevel)
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

func TestNewLinuxSeccompRemoveSet(t *testing.T) {
	tests := []struct {
		set   []string
		errno string

		expectedSet   []LinuxSeccompEntry
		expectedErrno LinuxSeccompErrno
		expectedErr   bool
	}{
		{
			[]string{"chmod", "chown"},
			"-EPERM",
			nil,
			"",
			true,
		},
		{
			[]string{"@appc.io/empty"},
			"EACCESS",
			[]LinuxSeccompEntry{"@appc.io/empty"},
			LinuxSeccompErrno("EACCESS"),
			false,
		},
		{
			[]string{"chmod", "chown"},
			"",
			[]LinuxSeccompEntry{"chmod", "chown"},
			LinuxSeccompErrno(""),
			false,
		},
		{
			[]string{},
			"",
			nil,
			"",
			true,
		},
	}
	for i, tt := range tests {
		c, err := NewLinuxSeccompRemoveSet(tt.errno, tt.set...)
		if tt.expectedErr {
			if err == nil {
				t.Errorf("#%d: did not get expected error", i)
			}
			continue
		}
		if gset := c.Set(); !reflect.DeepEqual(gset, tt.expectedSet) {
			t.Errorf("#%d: got set %#v, expected set %#v", i, gset, tt.expectedSet)
		}
		if gerrno := c.Errno(); !reflect.DeepEqual(gerrno, tt.expectedErrno) {
			t.Errorf("#%d: got errno %#v, expected errno %#v", i, gerrno, tt.expectedErrno)
		}
	}
}

func TestNewLinuxSeccompRetainSet(t *testing.T) {
	tests := []struct {
		set   []string
		errno string

		expectedSet   []LinuxSeccompEntry
		expectedErrno LinuxSeccompErrno
		expectedErr   bool
	}{
		{
			[]string{},
			"eaccess",
			nil,
			"",
			true,
		},
		{
			[]string{"chmod"},
			"EACCESS",
			[]LinuxSeccompEntry{"chmod"},
			LinuxSeccompErrno("EACCESS"),
			false,
		},
		{
			[]string{"chmod", "chown"},
			"",
			[]LinuxSeccompEntry{"chmod", "chown"},
			LinuxSeccompErrno(""),
			false,
		},
		{
			[]string{},
			"",
			nil,
			"",
			true,
		},
	}
	for i, tt := range tests {
		c, err := NewLinuxSeccompRetainSet(tt.errno, tt.set...)
		if tt.expectedErr {
			if err == nil {
				t.Errorf("#%d: did not get expected error", i)
			}
			continue
		}
		if gset := c.Set(); !reflect.DeepEqual(gset, tt.expectedSet) {
			t.Errorf("#%d: got set %#v, expected set %#v", i, gset, tt.expectedSet)
		}
		if gerrno := c.Errno(); !reflect.DeepEqual(gerrno, tt.expectedErrno) {
			t.Errorf("#%d: got errno %#v, expected errno %#v", i, gerrno, tt.expectedErrno)
		}
	}
}
