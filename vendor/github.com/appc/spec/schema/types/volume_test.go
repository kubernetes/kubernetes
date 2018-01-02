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

func bp(b bool) *bool {
	return &b
}

func sp(s string) *string {
	return &s
}

func ip(i int) *int {
	return &i
}

func TestVolumeToFromString(t *testing.T) {
	tests := []struct {
		s string
		v Volume
	}{
		{
			"foobar,kind=host,source=/tmp",
			Volume{
				Name:     "foobar",
				Kind:     "host",
				Source:   "/tmp",
				ReadOnly: nil,
				Mode:     nil,
				UID:      nil,
				GID:      nil,
			},
		},
		{
			"foobar,kind=host,source=/tmp,readOnly=false",
			Volume{
				Name:     "foobar",
				Kind:     "host",
				Source:   "/tmp",
				ReadOnly: bp(false),
				Mode:     nil,
				UID:      nil,
				GID:      nil,
			},
		},
		{
			"foobar,kind=host,source=/tmp,readOnly=true",
			Volume{
				Name:     "foobar",
				Kind:     "host",
				Source:   "/tmp",
				ReadOnly: bp(true),
				Mode:     nil,
				UID:      nil,
				GID:      nil,
			},
		},
		{
			"foobar,kind=empty",
			Volume{
				Name:     "foobar",
				Kind:     "empty",
				ReadOnly: nil,
				Mode:     sp(emptyVolumeDefaultMode),
				UID:      ip(emptyVolumeDefaultUID),
				GID:      ip(emptyVolumeDefaultGID),
			},
		},
		{
			"foobar,kind=empty,readOnly=true",
			Volume{
				Name:     "foobar",
				Kind:     "empty",
				ReadOnly: bp(true),
				Mode:     sp(emptyVolumeDefaultMode),
				UID:      ip(emptyVolumeDefaultUID),
				GID:      ip(emptyVolumeDefaultGID),
			},
		},
		{
			"foobar,kind=empty,readOnly=true,mode=0777",
			Volume{
				Name:     "foobar",
				Kind:     "empty",
				ReadOnly: bp(true),
				Mode:     sp("0777"),
				UID:      ip(emptyVolumeDefaultUID),
				GID:      ip(emptyVolumeDefaultGID),
			},
		},
		{
			"foobar,kind=empty,mode=0777,uid=1000",
			Volume{
				Name:     "foobar",
				Kind:     "empty",
				ReadOnly: nil,
				Mode:     sp("0777"),
				UID:      ip(1000),
				GID:      ip(emptyVolumeDefaultGID),
			},
		},
		{
			"foobar,kind=empty,mode=0777,uid=1000,gid=1000",
			Volume{
				Name:     "foobar",
				Kind:     "empty",
				ReadOnly: nil,
				Mode:     sp("0777"),
				UID:      ip(1000),
				GID:      ip(1000),
			},
		},
		{
			"foobar,kind=host,source=/tmp,recursive=false",
			Volume{
				Name:      "foobar",
				Kind:      "host",
				Source:    "/tmp",
				ReadOnly:  nil,
				Mode:      nil,
				UID:       nil,
				GID:       nil,
				Recursive: bp(false),
			},
		},
		{
			"foobar,kind=host,source=/tmp,recursive=true",
			Volume{
				Name:      "foobar",
				Kind:      "host",
				Source:    "/tmp",
				ReadOnly:  nil,
				Mode:      nil,
				UID:       nil,
				GID:       nil,
				Recursive: bp(true),
			},
		},
	}
	for i, tt := range tests {
		v, err := VolumeFromString(tt.s)
		if err != nil {
			t.Errorf("#%d: got err=%v, want nil", i, err)
		}
		if !reflect.DeepEqual(*v, tt.v) {
			t.Errorf("#%d: v=%v, want %v", i, *v, tt.v)
		}
		// volume serialization should be reversible
		o := v.String()
		if o != tt.s {
			t.Errorf("#%d: v.String()=%s, want %s", i, o, tt.s)
		}
	}
}

func TestVolumeFromStringBad(t *testing.T) {
	tests := []string{
		"#foobar,kind=host,source=/tmp",
		"foobar,kind=host,source=/tmp,readOnly=true,asdf=asdf",
		"foobar,kind=host,source=tmp",
		"foobar,kind=host,uid=3",
		"foobar,kind=host,mode=0755",
		"foobar,kind=host,mode=0600,readOnly=true,gid=0",
		"foobar,kind=empty,source=/tmp",
		"foobar,kind=host,source=/tmp,recursive=MAYBE",
	}
	for i, in := range tests {
		l, err := VolumeFromString(in)
		if l != nil {
			t.Errorf("#%d: got l=%v, want nil", i, l)
		}
		if err == nil {
			t.Errorf("#%d: got err=nil, want non-nil", i)
		}
	}
}

func BenchmarkVolumeToString(b *testing.B) {
	vol := Volume{
		Name:     "foobar",
		Kind:     "empty",
		ReadOnly: bp(true),
		Mode:     sp("0777"),
		UID:      ip(3),
		GID:      ip(4),
	}
	for i := 0; i < b.N; i++ {
		_ = vol.String()
	}
}
