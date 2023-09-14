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

package dump

import (
	"fmt"
	"testing"
)

func ptrint(i int) *int {
	return &i
}

func ptrstr(s string) *string {
	return &s
}

// custom type to test Stringer interface on non-pointer receiver.
type customString string

// String implements the Stringer interface for testing invocation
func (s customString) String() string {
	return "custom string " + string(s)
}

// custom type to test error interface on non-pointer receiver.
type customError int

// Error implements the error interface for testing invocation
func (e customError) Error() string {
	return fmt.Sprintf("custom error: %d", int(e))
}

// custom type to test Stringer interface on pointer receiver.
type pCustomString string

// String implements the Stringer interface for testing invocation
func (s *pCustomString) String() string {
	return "custom string " + string(*s)
}

// custom type to test error interface on pointer receiver.
type pCustomError int

// Error implements the error interface for testing invocation
func (e *pCustomError) Error() string {
	return fmt.Sprintf("custom error: %d", int(*e))
}

// embed is used to test embedded structures.
type embed struct {
	s string
}

// embedwrap is used to test embedded structures.
type embedwrap struct {
	embed
	e *embed
}

func TestPretty(t *testing.T) {
	tcs := customString("test")
	tpcs := pCustomString("&test")

	tce := customError(0)
	tpce := pCustomError(0)

	teb := embed{"test"}
	tebw := embedwrap{teb, &teb}

	testCases := []struct {
		a    interface{}
		want string
	}{
		{int8(93), "(int8) 93\n"},
		{int16(93), "(int16) 93\n"},
		{int32(93), "(int32) 93\n"},
		{int64(93), "(int64) 93\n"},
		{int(-93), "(int) -93\n"},
		{int8(-93), "(int8) -93\n"},
		{int16(-93), "(int16) -93\n"},
		{int32(-93), "(int32) -93\n"},
		{int64(-93), "(int64) -93\n"},
		{uint(93), "(uint) 93\n"},
		{uint8(93), "(uint8) 93\n"},
		{uint16(93), "(uint16) 93\n"},
		{uint32(93), "(uint32) 93\n"},
		{uint64(93), "(uint64) 93\n"},
		{uintptr(93), "(uintptr) 0x5d\n"},
		{ptrint(93), "(*int)(93)\n"},
		{float32(93.76), "(float32) 93.76\n"},
		{float64(93.76), "(float64) 93.76\n"},
		{complex64(93i), "(complex64) (0+93i)\n"},
		{complex128(93i), "(complex128) (0+93i)\n"},
		{bool(true), "(bool) true\n"},
		{bool(false), "(bool) false\n"},
		{string("test"), "(string) (len=4) \"test\"\n"},
		{ptrstr("test"), "(*string)((len=4) \"test\")\n"},
		{[1]string{"arr"}, "([1]string) (len=1) {\n  (string) (len=3) \"arr\"\n}\n"},
		{[]string{"slice"}, "([]string) (len=1) {\n  (string) (len=5) \"slice\"\n}\n"},
		{tcs, "(dump.customString) (len=4) \"test\"\n"},
		{&tcs, "(*dump.customString)((len=4) \"test\")\n"},
		{tpcs, "(dump.pCustomString) (len=5) \"&test\"\n"},
		{&tpcs, "(*dump.pCustomString)((len=5) \"&test\")\n"},
		{tce, "(dump.customError) 0\n"},
		{&tce, "(*dump.customError)(0)\n"},
		{tpce, "(dump.pCustomError) 0\n"},
		{&tpce, "(*dump.pCustomError)(0)\n"},
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
			"(struct { arr [1]string; slice []string; m map[string]int }) {\n  arr: ([1]string) (len=1) {\n    (string) (len=3) \"arr\"\n  },\n  slice: ([]string) (len=1) {\n    (string) (len=5) \"slice\"\n  },\n  m: (map[string]int) (len=1) {\n    (string) (len=3) \"one\": (int) 1\n  }\n}\n",
		},
		{teb, "(dump.embed) {\n  s: (string) (len=4) \"test\"\n}\n"},
		{tebw, "(dump.embedwrap) {\n  embed: (dump.embed) {\n    s: (string) (len=4) \"test\"\n  },\n  e: (*dump.embed)({\n    s: (string) (len=4) \"test\"\n  })\n}\n"},
		{map[string]int{}, "(map[string]int) {\n}\n"},
		{map[string]int{"one": 1}, "(map[string]int) (len=1) {\n  (string) (len=3) \"one\": (int) 1\n}\n"},
		{map[string]interface{}{"one": 1}, "(map[string]interface {}) (len=1) {\n  (string) (len=3) \"one\": (int) 1\n}\n"},
		{map[string]customString{"key": tcs}, "(map[string]dump.customString) (len=1) {\n  (string) (len=3) \"key\": (dump.customString) (len=4) \"test\"\n}\n"},
		{map[string]customError{"key": tce}, "(map[string]dump.customError) (len=1) {\n  (string) (len=3) \"key\": (dump.customError) 0\n}\n"},
		{map[string]embed{"key": teb}, "(map[string]dump.embed) (len=1) {\n  (string) (len=3) \"key\": (dump.embed) {\n    s: (string) (len=4) \"test\"\n  }\n}\n"},
		{map[string]embedwrap{"key": tebw}, "(map[string]dump.embedwrap) (len=1) {\n  (string) (len=3) \"key\": (dump.embedwrap) {\n    embed: (dump.embed) {\n      s: (string) (len=4) \"test\"\n    },\n    e: (*dump.embed)({\n      s: (string) (len=4) \"test\"\n    })\n  }\n}\n"},
	}

	for i, tc := range testCases {
		s := Pretty(tc.a)
		if tc.want != s {
			t.Errorf("[%d]:\n\texpected %q\n\tgot      %q", i, tc.want, s)
		}
	}
}

func TestForHash(t *testing.T) {
	tcs := customString("test")
	tpcs := pCustomString("&test")

	tce := customError(0)
	tpce := pCustomError(0)

	teb := embed{"test"}
	tebw := embedwrap{teb, &teb}

	testCases := []struct {
		a    interface{}
		want string
	}{
		{int8(93), "(int8)93"},
		{int16(93), "(int16)93"},
		{int32(93), "(int32)93"},
		{int64(93), "(int64)93"},
		{int(-93), "(int)-93"},
		{int8(-93), "(int8)-93"},
		{int16(-93), "(int16)-93"},
		{int32(-93), "(int32)-93"},
		{int64(-93), "(int64)-93"},
		{uint(93), "(uint)93"},
		{uint8(93), "(uint8)93"},
		{uint16(93), "(uint16)93"},
		{uint32(93), "(uint32)93"},
		{uint64(93), "(uint64)93"},
		{uintptr(93), "(uintptr)0x5d"},
		{ptrint(93), "(*int)93"},
		{float32(93.76), "(float32)93.76"},
		{float64(93.76), "(float64)93.76"},
		{complex64(93i), "(complex64)(0+93i)"},
		{complex128(93i), "(complex128)(0+93i)"},
		{bool(true), "(bool)true"},
		{bool(false), "(bool)false"},
		{string("test"), "(string)test"},
		{ptrstr("test"), "(*string)test"},
		{[1]string{"arr"}, "([1]string)[arr]"},
		{[]string{"slice"}, "([]string)[slice]"},
		{tcs, "(dump.customString)test"},
		{&tcs, "(*dump.customString)test"},
		{tpcs, "(dump.pCustomString)&test"},
		{&tpcs, "(*dump.pCustomString)&test"},
		{tce, "(dump.customError)0"},
		{&tce, "(*dump.customError)0"},
		{tpce, "(dump.pCustomError)0"},
		{&tpce, "(*dump.pCustomError)0"},
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
			"(struct { arr [1]string; slice []string; m map[string]int }){arr:([1]string)[arr] slice:([]string)[slice] m:(map[string]int)map[one:1]}",
		},
		{teb, "(dump.embed){s:(string)test}"},
		{tebw, "(dump.embedwrap){embed:(dump.embed){s:(string)test} e:(*dump.embed){s:(string)test}}"},
		{map[string]int{}, "(map[string]int)map[]"},
		{map[string]int{"one": 1, "two": 2}, "(map[string]int)map[one:1 two:2]"},
		{map[string]interface{}{"one": 1}, "(map[string]interface {})map[one:(int)1]"},
		{map[string]customString{"key": tcs}, "(map[string]dump.customString)map[key:test]"},
		{map[string]customError{"key": tce}, "(map[string]dump.customError)map[key:0]"},
		{map[string]embed{"key": teb}, "(map[string]dump.embed)map[key:{s:(string)test}]"},
		{map[string]embedwrap{"key": tebw}, "(map[string]dump.embedwrap)map[key:{embed:(dump.embed){s:(string)test} e:(*dump.embed){s:(string)test}}]"},
	}

	for i, tc := range testCases {
		s := ForHash(tc.a)
		if tc.want != s {
			t.Errorf("[%d]:\n\texpected %q\n\tgot      %q", i, tc.want, s)
		}
	}
}

func TestOneLine(t *testing.T) {
	tcs := customString("test")
	tpcs := pCustomString("&test")

	tce := customError(0)
	tpce := pCustomError(0)

	teb := embed{"test"}
	tebw := embedwrap{teb, &teb}

	testCases := []struct {
		a    interface{}
		want string
	}{
		{int8(93), "(int8)93"},
		{int16(93), "(int16)93"},
		{int32(93), "(int32)93"},
		{int64(93), "(int64)93"},
		{int(-93), "(int)-93"},
		{int8(-93), "(int8)-93"},
		{int16(-93), "(int16)-93"},
		{int32(-93), "(int32)-93"},
		{int64(-93), "(int64)-93"},
		{uint(93), "(uint)93"},
		{uint8(93), "(uint8)93"},
		{uint16(93), "(uint16)93"},
		{uint32(93), "(uint32)93"},
		{uint64(93), "(uint64)93"},
		{uintptr(93), "(uintptr)0x5d"},
		{ptrint(93), "(*int)93"},
		{float32(93.76), "(float32)93.76"},
		{float64(93.76), "(float64)93.76"},
		{complex64(93i), "(complex64)(0+93i)"},
		{complex128(93i), "(complex128)(0+93i)"},
		{bool(true), "(bool)true"},
		{bool(false), "(bool)false"},
		{string("test"), "(string)test"},
		{ptrstr("test"), "(*string)test"},
		{[1]string{"arr"}, "([1]string)[arr]"},
		{[]string{"slice"}, "([]string)[slice]"},
		{tcs, "(dump.customString)test"},
		{&tcs, "(*dump.customString)test"},
		{tpcs, "(dump.pCustomString)&test"},
		{&tpcs, "(*dump.pCustomString)&test"},
		{tce, "(dump.customError)0"},
		{&tce, "(*dump.customError)0"},
		{tpce, "(dump.pCustomError)0"},
		{&tpce, "(*dump.pCustomError)0"},
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
			"(struct { arr [1]string; slice []string; m map[string]int }){arr:([1]string)[arr] slice:([]string)[slice] m:(map[string]int)map[one:1]}",
		},
		{teb, "(dump.embed){s:(string)test}"},
		{tebw, "(dump.embedwrap){embed:(dump.embed){s:(string)test} e:(*dump.embed){s:(string)test}}"},
		{map[string]int{}, "(map[string]int)map[]"},
		{map[string]int{"one": 1}, "(map[string]int)map[one:1]"},
		{map[string]interface{}{"one": 1}, "(map[string]interface {})map[one:(int)1]"},
		{map[string]customString{"key": tcs}, "(map[string]dump.customString)map[key:test]"},
		{map[string]customError{"key": tce}, "(map[string]dump.customError)map[key:0]"},
		{map[string]embed{"key": teb}, "(map[string]dump.embed)map[key:{s:(string)test}]"},
		{map[string]embedwrap{"key": tebw}, "(map[string]dump.embedwrap)map[key:{embed:(dump.embed){s:(string)test} e:(*dump.embed){s:(string)test}}]"},
	}

	for i, tc := range testCases {
		s := OneLine(tc.a)
		if tc.want != s {
			t.Errorf("[%d]:\n\texpected %q\n\tgot      %q", i, tc.want, s)
		}
	}
}
