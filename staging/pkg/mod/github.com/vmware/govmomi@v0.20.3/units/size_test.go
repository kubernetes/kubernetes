/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package units

import (
	"fmt"
	"math"
	"testing"
)

func TestMB(t *testing.T) {
	b := ByteSize(1024 * 1024)
	expected := "1.0MB"
	if b.String() != expected {
		t.Errorf("Expected '%v' but got '%v'", expected, b)
	}
}

func TestTenMB(t *testing.T) {
	b := ByteSize(10 * 1024 * 1024)
	actual := fmt.Sprintf("%s", b)
	expected := "10.0MB"
	if actual != expected {
		t.Errorf("Expected '%v' but got '%v'", expected, actual)
	}
}

func assertEquals(t *testing.T, expected string, actual ByteSize) {
	if expected != actual.String() {
		t.Errorf("Expected '%v' but got '%v'", expected, actual.String())
	}
}

func TestByteSize(t *testing.T) {
	assertEquals(t, "1B", ByteSize(1))
	assertEquals(t, "10B", ByteSize(10))
	assertEquals(t, "100B", ByteSize(100))
	assertEquals(t, "1000B", ByteSize(1000))
	assertEquals(t, "1.0KB", ByteSize(1024))
	assertEquals(t, "1.0MB", ByteSize(1024*1024))
	assertEquals(t, "1.0MB", ByteSize(1048576))
	assertEquals(t, "10.0MB", ByteSize(10*math.Pow(1024, 2)))
	assertEquals(t, "1.0GB", ByteSize(1024*1024*1024))
	assertEquals(t, "1.0TB", ByteSize(1024*1024*1024*1024))
	assertEquals(t, "1.0TB", ByteSize(1048576*1048576))
	assertEquals(t, "1.0PB", ByteSize(1024*1024*1024*1024*1024))
	assertEquals(t, "1.0EB", ByteSize(1024*1024*1024*1024*1024*1024))
	assertEquals(t, "1.0EB", ByteSize(1048576*1048576*1048576))
}

func TestByteSizeSet(t *testing.T) {
	var tests = []struct {
		In     string
		OutStr string
		Out    ByteSize
	}{
		{
			In:     "345",
			OutStr: "345B",
			Out:    345.0,
		},
		{
			In:     "345b",
			OutStr: "345B",
			Out:    345.0,
		},
		{
			In:     "345K",
			OutStr: "345.0KB",
			Out:    345 * KB,
		},
		{
			In:     "345kb",
			OutStr: "345.0KB",
			Out:    345 * KB,
		},
		{
			In:     "345kib",
			OutStr: "345.0KB",
			Out:    345 * KB,
		},
		{
			In:     "345KiB",
			OutStr: "345.0KB",
			Out:    345 * KB,
		},
		{
			In:     "345M",
			OutStr: "345.0MB",
			Out:    345 * MB,
		},
		{
			In:     "345G",
			OutStr: "345.0GB",
			Out:    345 * GB,
		},
		{
			In:     "345T",
			OutStr: "345.0TB",
			Out:    345 * TB,
		},
		{
			In:     "345P",
			OutStr: "345.0PB",
			Out:    345 * PB,
		},
		{
			In:     "3E",
			OutStr: "3.0EB",
			Out:    3 * EB,
		},
	}

	for _, test := range tests {
		var v ByteSize
		err := v.Set(test.In)
		if err != nil {
			t.Errorf("Error: %s [%v]", err, test.In)
			continue
		}

		if v != test.Out {
			t.Errorf("Out: expect '%v' actual '%v'", test.Out, v)
			continue
		}

		if v.String() != test.OutStr {
			t.Errorf("String: expect '%v' actual '%v'", test.OutStr, v.String())
		}
	}
}
