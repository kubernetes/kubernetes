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

package print

import (
	"testing"
)

func ptrint(i int) *int {
	return &i
}

func ptrstr(s string) *string {
	return &s
}

func TestPrettyPrintObject(t *testing.T) {
	testCases := []struct {
		a        interface{}
		withType bool
		want     string
	}{
		{int8(93), true, "(int8)93"},
		{int16(93), true, "(int16)93"},
		{int32(93), true, "(int32)93"},
		{int64(93), true, "(int64)93"},
		{int(-93), true, "(int)-93"},
		{int8(-93), true, "(int8)-93"},
		{int16(-93), true, "(int16)-93"},
		{int32(-93), true, "(int32)-93"},
		{int64(-93), true, "(int64)-93"},
		{uint(93), true, "(uint)93"},
		{uint8(93), true, "(uint8)93"},
		{uint16(93), true, "(uint16)93"},
		{uint32(93), true, "(uint32)93"},
		{uint64(93), true, "(uint64)93"},
		{uintptr(93), true, "(uintptr)0x5d"},
		{ptrint(93), true, "(*int)93"},
		{float32(93.76), true, "(float32)93.76"},
		{float64(93.76), true, "(float64)93.76"},
		{complex64(93i), true, "(complex64)(0+93i)"},
		{complex128(93i), true, "(complex128)(0+93i)"},
		{bool(true), true, "(bool)true"},
		{bool(false), true, "(bool)false"},
		{string("test"), true, "(string)test"},
		{ptrstr("test"), true, "(*string)test"},
		{
			struct {
				arr   [1]string
				slice []string
				m     map[string]int
			}{
				[1]string{"arr"},
				[]string{"slice"},
				map[string]int{"one": 1},
			},
			true,
			"(struct { arr [1]string; slice []string; m map[string]int }){arr:([1]string)[arr] slice:([]string)[slice] m:(map[string]int)map[one:1]}",
		},
		{int8(93), false, "93"},
		{int16(93), false, "93"},
		{int32(93), false, "93"},
		{int64(93), false, "93"},
		{int(-93), false, "-93"},
		{int8(-93), false, "-93"},
		{int16(-93), false, "-93"},
		{int32(-93), false, "-93"},
		{int64(-93), false, "-93"},
		{uint(93), false, "93"},
		{uint8(93), false, "93"},
		{uint16(93), false, "93"},
		{uint32(93), false, "93"},
		{uint64(93), false, "93"},
		{uintptr(93), false, "0x5d"},
		{float32(93.76), false, "93.76"},
		{float64(93.76), false, "93.76"},
		{complex64(93i), false, "(0+93i)"},
		{complex128(93i), false, "(0+93i)"},
		{bool(true), false, "true"},
		{bool(false), false, "false"},
		{string("test"), false, "test"},
		{
			struct {
				arr   [1]string
				slice []string
				m     map[string]int
			}{
				[1]string{"arr"},
				[]string{"slice"},
				map[string]int{"one": 1},
			},
			false,
			"{arr:[arr] slice:[slice] m:map[one:1]}",
		},
	}

	for i, tc := range testCases {
		s := PrettyPrintObject(tc.a, tc.withType)
		if tc.want != s {
			t.Errorf("[%d]:\n\texpected %q\n\tgot      %q", i, tc.want, s)
		}
	}
}
