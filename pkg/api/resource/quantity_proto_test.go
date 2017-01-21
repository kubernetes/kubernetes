/*
Copyright 2017 The Kubernetes Authors.

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

package resource

import (
	"bytes"
	"testing"

	inf "gopkg.in/inf.v0"
)

func TestQuantityProtoMarshal(t *testing.T) {
	table := []struct {
		quantity string
		expect   []byte
	}{
		{"0", []byte([]uint8{10, 1, 48})},
		{"100m", []byte([]uint8{10, 4, 49, 48, 48, 109})},
		{"50m", []byte([]uint8{10, 3, 53, 48, 109})},
		{"10000T", []byte([]uint8{10, 3, 49, 48, 80})},
	}
	for _, testCase := range table {
		q := MustParse(testCase.quantity)
		// Won't currently get an error as MarshalTo can't return one
		result, _ := q.Marshal()
		if !bytes.Equal(result, testCase.expect) {
			t.Errorf("q: %v, Expected: %v, Actual: %v", q, testCase.expect, result)
		}
	}

	nils := []struct {
		dec    *inf.Dec
		expect []byte
	}{
		{dec(0, 0).Dec, []byte([]uint8{10, 1, 48})},
		{dec(10, 0).Dec, []byte([]uint8{10, 2, 49, 48})},
		{dec(-10, 0).Dec, []byte([]uint8{10, 3, 45, 49, 48})},
	}
	for _, nilCase := range nils {
		q := Quantity{d: infDecAmount{nilCase.dec}, Format: DecimalSI}
		// Won't currently get an error as MarshalTo can't return one
		result, _ := q.Marshal()
		if !bytes.Equal(result, nilCase.expect) {
			t.Errorf("q: %v, Expected: %v, Actual: %v", q, nilCase.expect, result)
		}
	}
}

func TestQuantityProtoUnmarshal(t *testing.T) {
	table := []struct {
		input  []byte
		expect string
	}{
		{[]byte([]uint8{10, 1, 48}), "0"},
		{[]byte([]uint8{10, 4, 49, 48, 48, 109}), "100m"},
		{[]byte([]uint8{10, 3, 53, 48, 109}), "50m"},
		{[]byte([]uint8{10, 3, 49, 48, 80}), "10000T"},
	}
	for _, testCase := range table {
		var q Quantity
		q.Unmarshal(testCase.input)
		e := MustParse(testCase.expect)
		if q.Cmp(e) != 0 {
			t.Errorf("Expected: %v, Actual: %v", e, q)
		}
	}

	nils := []struct {
		input  []byte
		expect *inf.Dec
	}{
		{[]byte([]uint8{10, 1, 48}), dec(0, 0).Dec},
		{[]byte([]uint8{10, 2, 49, 48}), dec(10, 0).Dec},
		{[]byte([]uint8{10, 3, 45, 49, 48}), dec(-10, 0).Dec},
	}
	for _, nilCase := range nils {
		var q Quantity
		q.Unmarshal(nilCase.input)
		e := Quantity{d: infDecAmount{nilCase.expect}, Format: DecimalSI}
		if q.Cmp(e) != 0 {
			t.Errorf("Expected: %v, Actual: %v", e, q)
		}
	}
}
