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

func TestVolumeFromString(t *testing.T) {
	trueVar := true
	falseVar := false
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
			},
		},
		{
			"foobar,kind=host,source=/tmp,readOnly=false",
			Volume{
				Name:     "foobar",
				Kind:     "host",
				Source:   "/tmp",
				ReadOnly: &falseVar,
			},
		},
		{
			"foobar,kind=host,source=/tmp,readOnly=true",
			Volume{
				Name:     "foobar",
				Kind:     "host",
				Source:   "/tmp",
				ReadOnly: &trueVar,
			},
		},
		{
			"foobar,kind=empty",
			Volume{
				Name:     "foobar",
				Kind:     "empty",
				ReadOnly: nil,
			},
		},
		{
			"foobar,kind=empty,readOnly=true",
			Volume{
				Name:     "foobar",
				Kind:     "empty",
				ReadOnly: &trueVar,
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
	}
}

func TestVolumeFromStringBad(t *testing.T) {
	tests := []string{
		"#foobar,kind=host,source=/tmp",
		"foobar,kind=host,source=/tmp,readOnly=true,asdf=asdf",
		"foobar,kind=empty,source=/tmp",
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
